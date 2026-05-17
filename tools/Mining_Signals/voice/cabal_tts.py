"""CABAL-style TTS: Piper (offline neural TTS) → effects chain → audio.

No LLM, no cloud. One-core CPU burst per utterance, zero idle cost.

The "CABAL" character is entirely in the post-processing — Piper gives
a clean deep male voice, then we pitch it down, ring-modulate it for
the metallic rasp, bit-crush + light distortion for digital grit, and
add a short delay + plate reverb for that "speaking from inside a
machine" space. Swap voice models or tweak ``EffectsConfig`` to taste.

First call downloads the voice model (~60 MB) into
``~/.cache/piper/`` and caches it forever. Subsequent calls reuse it.

Dependencies (install once)::

    pip install piper-tts pedalboard sounddevice numpy

``piper-tts`` bundles the ONNX runtime; no separate install needed.
``sounddevice`` is only used by :func:`speak`; :func:`synthesize` and
:func:`save_wav` work without it.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Voice model catalogue
# ─────────────────────────────────────────────────────────────
#
# Piper publishes voice models as (.onnx + .onnx.json) pairs on
# HuggingFace. We default to ``alan-medium`` — a deep British male
# voice that processes well into the CABAL timbre. Override via
# ``VoiceConfig(model="...")`` if you want a different base.
#
# Catalogue of useful CABAL-base candidates (all English, male,
# lower register):
#
#   en_GB-alan-medium       deep British, slightly flat → good base
#   en_US-lessac-medium     American, neutral, very clean
#   en_US-ryan-medium       American, warmer
#   en_GB-northern_english_male-medium   regional, more character
#
# Full catalogue: https://github.com/rhasspy/piper/blob/master/VOICES.md

_PIPER_HF_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
)

_VOICE_URLS = {
    "en_GB-alan-medium": (
        _PIPER_HF_BASE + "en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        _PIPER_HF_BASE + "en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
    ),
    "en_US-lessac-medium": (
        _PIPER_HF_BASE + "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        _PIPER_HF_BASE + "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    ),
    "en_US-ryan-medium": (
        _PIPER_HF_BASE + "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        _PIPER_HF_BASE + "en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    ),
}


@dataclass
class VoiceConfig:
    """Which Piper voice to use and where to cache it."""
    model: str = "en_GB-alan-medium"
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "piper",
    )
    # Piper synthesis knobs. ``length_scale`` > 1 slows speech; < 1
    # speeds it up. CABAL speaks deliberately so we slow slightly.
    length_scale: float = 1.08
    noise_scale: float = 0.667
    noise_w: float = 0.8


@dataclass
class EffectsConfig:
    """Effects chain parameters. Defaults are tuned for CABAL-ish.

    Set any stage's ``*_enabled`` to False to bypass that stage while
    keeping the rest of the chain intact — useful for A/B tuning.
    """
    # Pitch shift (semitones). Negative = lower.
    pitch_semitones: float = -2.5
    pitch_enabled: bool = True

    # Ring modulator frequency (Hz). 20–40 Hz gives the classic
    # "robotic rasp"; 80+ Hz becomes more alien/buzzy.
    ringmod_hz: float = 30.0
    ringmod_mix: float = 0.35          # 0=dry, 1=full ringmod
    ringmod_enabled: bool = True

    # Bit crusher — reduces bit depth for digital grit.
    bit_depth: int = 10                 # 16 = clean, 6 = very crunchy
    bitcrush_enabled: bool = True

    # Soft distortion drive (dB).
    distortion_db: float = 6.0
    distortion_enabled: bool = True

    # Short slap delay (ms).
    delay_ms: float = 85.0
    delay_feedback: float = 0.22
    delay_mix: float = 0.25
    delay_enabled: bool = True

    # Plate-style reverb.
    reverb_size: float = 0.55           # 0=tight, 1=cavernous
    reverb_damping: float = 0.35
    reverb_wet: float = 0.22
    reverb_dry: float = 0.78
    reverb_enabled: bool = True

    # Final output gain (dB) — compensate for delay+reverb adding
    # energy. Usually slightly negative.
    output_gain_db: float = -1.5


# ─────────────────────────────────────────────────────────────
# Piper loader (lazy, thread-safe)
# ─────────────────────────────────────────────────────────────

_piper_lock = threading.Lock()
_piper_voice = None           # cached PiperVoice instance
_piper_sr: int | None = None  # its sample rate


def _ensure_model(cfg: VoiceConfig) -> tuple[Path, Path]:
    """Return (onnx_path, json_path), downloading if missing."""
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    onnx = cfg.cache_dir / f"{cfg.model}.onnx"
    meta = cfg.cache_dir / f"{cfg.model}.onnx.json"
    if onnx.exists() and meta.exists():
        return onnx, meta

    urls = _VOICE_URLS.get(cfg.model)
    if not urls:
        raise ValueError(
            f"Unknown voice {cfg.model!r}. Known: {list(_VOICE_URLS)}. "
            "For other voices, drop the .onnx/.onnx.json pair into "
            f"{cfg.cache_dir} manually."
        )
    for url, dest in zip(urls, (onnx, meta)):
        if dest.exists():
            continue
        log.info("piper: downloading %s → %s", url, dest)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
        tmp.rename(dest)
    return onnx, meta


def _get_voice(cfg: VoiceConfig):
    global _piper_voice, _piper_sr
    with _piper_lock:
        if _piper_voice is not None:
            return _piper_voice, _piper_sr
        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "piper-tts is not installed. Run: "
                "pip install piper-tts pedalboard sounddevice"
            ) from exc
        onnx, _meta = _ensure_model(cfg)
        _piper_voice = PiperVoice.load(str(onnx))
        _piper_sr = _piper_voice.config.sample_rate
        log.info("piper: loaded %s @ %d Hz", cfg.model, _piper_sr)
        return _piper_voice, _piper_sr


# ─────────────────────────────────────────────────────────────
# Raw synthesis (Piper → float32 mono numpy)
# ─────────────────────────────────────────────────────────────

def _piper_synth(text: str, cfg: VoiceConfig) -> tuple[np.ndarray, int]:
    voice, sr = _get_voice(cfg)

    # Piper streams int16 PCM into a buffer; we assemble then convert.
    # ``synthesize_wav`` writes a full WAV (with header) into a file
    # handle. We use BytesIO to keep it in memory.
    #
    # Current piper-tts wraps the synthesis knobs in a SynthesisConfig
    # object rather than accepting them as top-level kwargs. Using the
    # old kwarg signature raises TypeError, which left the wave buffer
    # empty and produced a confusing "# channels not specified" error
    # from the read-back open below.
    try:
        from piper.config import SynthesisConfig  # type: ignore
    except Exception:
        SynthesisConfig = None  # type: ignore[assignment]
    syn_cfg = None
    if SynthesisConfig is not None:
        syn_cfg = SynthesisConfig(
            length_scale=cfg.length_scale,
            noise_scale=cfg.noise_scale,
            noise_w_scale=cfg.noise_w,
        )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        if syn_cfg is not None:
            voice.synthesize_wav(text, wf, syn_config=syn_cfg)
        else:
            voice.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        # Piper's reported sample rate may disagree with the voice's
        # cached value (newer models can override at synthesis time);
        # trust what the wave header says.
        sr = wf.getframerate()
    return audio, sr


# ─────────────────────────────────────────────────────────────
# Effects chain
# ─────────────────────────────────────────────────────────────

def _ring_modulate(audio: np.ndarray, sr: int, hz: float, mix: float) -> np.ndarray:
    """Multiply by a sine wave → classic robotic timbre.

    Pure numpy; no external DSP lib needed. ``mix`` blends the
    ringmod'd signal with the dry signal (0=dry, 1=full wet).
    """
    t = np.arange(len(audio), dtype=np.float32) / sr
    carrier = np.sin(2.0 * np.pi * hz * t).astype(np.float32)
    wet = audio * carrier
    return (1.0 - mix) * audio + mix * wet


def _apply_effects(
    audio: np.ndarray,
    sr: int,
    fx: EffectsConfig,
) -> np.ndarray:
    """Run the dry Piper output through the CABAL effects chain."""
    try:
        from pedalboard import (  # type: ignore
            Pedalboard, PitchShift, Bitcrush, Distortion, Delay, Reverb, Gain,
        )
    except ImportError as exc:
        raise RuntimeError(
            "pedalboard is not installed. Run: pip install pedalboard"
        ) from exc

    stages = []
    if fx.pitch_enabled and fx.pitch_semitones != 0.0:
        stages.append(PitchShift(semitones=fx.pitch_semitones))

    # Ring mod is done by hand before the pedalboard chain because
    # pedalboard has no built-in ringmod plugin.
    if fx.ringmod_enabled and fx.ringmod_mix > 0.0:
        audio = _ring_modulate(audio, sr, fx.ringmod_hz, fx.ringmod_mix)

    if fx.bitcrush_enabled:
        stages.append(Bitcrush(bit_depth=fx.bit_depth))
    if fx.distortion_enabled and fx.distortion_db > 0:
        stages.append(Distortion(drive_db=fx.distortion_db))
    if fx.delay_enabled and fx.delay_mix > 0:
        stages.append(Delay(
            delay_seconds=fx.delay_ms / 1000.0,
            feedback=fx.delay_feedback,
            mix=fx.delay_mix,
        ))
    if fx.reverb_enabled and fx.reverb_wet > 0:
        stages.append(Reverb(
            room_size=fx.reverb_size,
            damping=fx.reverb_damping,
            wet_level=fx.reverb_wet,
            dry_level=fx.reverb_dry,
        ))
    stages.append(Gain(gain_db=fx.output_gain_db))

    board = Pedalboard(stages)
    # Pedalboard expects float32 shape (channels, samples); we're mono.
    processed = board(audio[np.newaxis, :], sr)[0]

    # Safety limit — prevent effect stacking from clipping the DAC.
    peak = np.max(np.abs(processed))
    if peak > 0.99:
        processed = processed * (0.99 / peak)
    return processed.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def synthesize(
    text: str,
    *,
    voice: Optional[VoiceConfig] = None,
    effects: Optional[EffectsConfig] = None,
) -> tuple[np.ndarray, int]:
    """Synthesize ``text`` → (float32 mono audio, sample_rate).

    Pure function; doesn't touch the sound card. Use for saving,
    mixing, or piping into another audio system.
    """
    vcfg = voice or VoiceConfig()
    fcfg = effects or EffectsConfig()
    dry, sr = _piper_synth(text, vcfg)
    wet = _apply_effects(dry, sr, fcfg)
    return wet, sr


def save_wav(
    text: str,
    path: str | os.PathLike,
    *,
    voice: Optional[VoiceConfig] = None,
    effects: Optional[EffectsConfig] = None,
) -> None:
    """Synthesize and write a 16-bit mono WAV to ``path``."""
    audio, sr = synthesize(text, voice=voice, effects=effects)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def speak(
    text: str,
    *,
    voice: Optional[VoiceConfig] = None,
    effects: Optional[EffectsConfig] = None,
    blocking: bool = False,
) -> None:
    """Synthesize and play through the default output device.

    ``blocking=False`` (default) returns immediately; synthesis runs
    on a worker thread and playback starts as soon as audio is ready.
    Subsequent ``speak`` calls while audio is still playing will
    overlap — wrap in your own queue if you need sequential output.
    """
    def _run():
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            log.error(
                "sounddevice not installed — cannot play audio. "
                "Install with: pip install sounddevice. "
                "(synthesize() and save_wav() work without it.)",
            )
            return
        try:
            audio, sr = synthesize(text, voice=voice, effects=effects)
            sd.play(audio, sr)
            if blocking:
                sd.wait()
        except Exception:
            log.exception("cabal_tts: speak() failed")

    if blocking:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="CABAL-style TTS smoke test")
    ap.add_argument("text", nargs="?",
                    default="Scan complete. Aluminum ore detected. "
                            "Mass five thousand six hundred seven kilograms. "
                            "Rock is breakable.")
    ap.add_argument("--save", type=str, default=None,
                    help="Write output to WAV instead of playing")
    ap.add_argument("--dry", action="store_true",
                    help="Bypass effects chain (hear raw Piper voice)")
    ap.add_argument("--model", default="en_GB-alan-medium",
                    help="Piper voice model name")
    args = ap.parse_args()

    vcfg = VoiceConfig(model=args.model)
    fcfg = EffectsConfig()
    if args.dry:
        for k in list(fcfg.__dict__):
            if k.endswith("_enabled"):
                setattr(fcfg, k, False)

    if args.save:
        save_wav(args.text, args.save, voice=vcfg, effects=fcfg)
        print(f"wrote {args.save}")
    else:
        speak(args.text, voice=vcfg, effects=fcfg, blocking=True)
