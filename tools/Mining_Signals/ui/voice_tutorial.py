"""CABAL voice tutorial — pre-recorded WAV with one-time TTS bootstrap.

Strategy:
  * If ``assets/audio/calibration_tutorial.wav`` exists, play it directly.
    No network call, no TTS service required. Instant playback.
  * If it does NOT exist, synthesize it ONCE via local Pocket TTS
    (port 49112) and save permanently to the project. Future clicks
    replay the saved file.

So the user only needs Pocket TTS running for the FIRST playback to
generate the asset. After that the WAV ships with the project and
always works.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

POCKET_TTS_URL = "http://localhost:49112/tts"

# Permanent location for the pre-recorded tutorial audio. Lives in the
# project so it ships with installs / git checkpoints. Once generated,
# it's reused forever (no per-session synthesis).
#
# We support either .wav or .mp3. ElevenLabs returns mp3 by default;
# Pocket TTS returns wav. Qt's QMediaPlayer handles both natively.
_THIS_DIR = Path(__file__).resolve().parent
_AUDIO_DIR = _THIS_DIR.parent / "assets" / "audio"
_TUTORIAL_BASENAME = "calibration_tutorial"
TUTORIAL_WAV_PATH = _AUDIO_DIR / f"{_TUTORIAL_BASENAME}.wav"
TUTORIAL_MP3_PATH = _AUDIO_DIR / f"{_TUTORIAL_BASENAME}.mp3"


def _find_cached_tutorial() -> Optional[Path]:
    """Return the cached audio path if it exists in any supported
    format (mp3 preferred, then wav)."""
    if TUTORIAL_MP3_PATH.is_file():
        return TUTORIAL_MP3_PATH
    if TUTORIAL_WAV_PATH.is_file():
        return TUTORIAL_WAV_PATH
    return None

# CABAL voice walkthrough — phrased for slow, dramatic delivery.
TUTORIAL_SCRIPT = """\
Welcome, miner. Let me walk you through calibrating the mining HUD scanner.

Step one. Aim your mining laser at any rock. The SCAN RESULTS panel
should appear on your screen.

Step two. Look at the CALIBRATE tab. Each row shows the live crop being
fed to the optical character recognizer. The crops update every fraction
of a second.

Step three. Check that the value digits are fully visible inside each
row's preview. Mass. Resistance. Instability.

Step four. If a crop is misaligned, use the arrow buttons to nudge it.
Left, right, up, down. Hold shift while clicking to move five pixels at
a time. Use the W and H buttons to resize.

Step five. When a row's crop looks correct, click the LOCK button. The
row turns green. The coordinates are saved immediately to disk.

Step six. Repeat for all three required rows. Mass. Resistance.
Instability.

When all three are locked, the dialog displays "CALIBRATION COMPLETE"
in large text. You may close the dialog.

From this moment forward, the optical character recognizer will use
your confirmed coordinates instead of auto-detecting. No drift. No
edge cases. Maximum accuracy.

Good luck out there, miner.
"""


def is_available() -> bool:
    """Best-effort check whether Pocket TTS is reachable."""
    try:
        import requests
        r = requests.get(
            POCKET_TTS_URL.replace("/tts", "/"),
            timeout=0.5,
        )
        return True  # any HTTP response means service is up
    except Exception:
        return False


def get_tutorial_audio() -> tuple[Optional[Path], str]:
    """Return the path to the calibration tutorial audio file.

    If the pre-recorded WAV already exists at TUTORIAL_WAV_PATH, return it
    immediately. Otherwise synthesize it ONCE via local Pocket TTS and
    save it to that path so subsequent calls are instant.

    Returns ``(path, source)`` where source is one of:
        "cached"    — file already existed
        "generated" — just synthesized via Pocket TTS
        "missing"   — could not generate (Pocket TTS unreachable),
                      path is None
    """
    cached = _find_cached_tutorial()
    if cached is not None:
        return cached, "cached"
    # Not cached — generate via Pocket TTS one time
    log.info("voice_tutorial: cached audio missing, generating via Pocket TTS")
    try:
        import requests
    except ImportError:
        log.warning("requests not installed — voice tutorial disabled")
        return None, "missing"
    try:
        resp = requests.post(
            POCKET_TTS_URL,
            data={"text": TUTORIAL_SCRIPT},
            timeout=120.0,  # generation can take a while for long text
        )
    except Exception as exc:
        log.warning("voice_tutorial: Pocket TTS unreachable: %s", exc)
        return None, "missing"
    if resp.status_code != 200:
        log.warning(
            "voice_tutorial: Pocket TTS returned %d: %s",
            resp.status_code, resp.text[:200],
        )
        return None, "missing"
    try:
        TUTORIAL_WAV_PATH.parent.mkdir(parents=True, exist_ok=True)
        TUTORIAL_WAV_PATH.write_bytes(resp.content)
        log.info(
            "voice_tutorial: cached %d bytes to %s",
            len(resp.content), TUTORIAL_WAV_PATH,
        )
        return TUTORIAL_WAV_PATH, "generated"
    except Exception as exc:
        log.warning("voice_tutorial: save failed: %s", exc)
        return None, "missing"


def regenerate_tutorial_audio() -> tuple[Optional[Path], str]:
    """Force re-generation (e.g., if the script was updated).
    Deletes any cached audio and triggers a fresh synthesis."""
    for p in (TUTORIAL_WAV_PATH, TUTORIAL_MP3_PATH):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return get_tutorial_audio()


class VoicePlayer:
    """Wraps Qt's QMediaPlayer with simpler play/stop semantics.

    Owns the player object so it isn't garbage-collected mid-playback.
    """

    def __init__(self, on_state_change: Optional[Callable[[str], None]] = None):
        self._on_state_change = on_state_change
        self._player = None
        self._audio_output = None

    def _ensure_player(self) -> bool:
        if self._player is not None:
            return True
        # Qt6's default media backend on Windows is FFmpeg, which
        # fails on some mp3 streams with "# channels not specified"
        # (the error we've been seeing in the calibration dialog).
        # The native Windows Media Foundation backend handles mp3
        # cleanly out of the box. Force it BEFORE the QtMultimedia
        # import resolves — QT_MEDIA_BACKEND is read during the first
        # QMediaPlayer construction, so setting it here is early
        # enough as long as the player hasn't been built yet (we're
        # guarded by `self._player is not None` above).
        if os.name == "nt" and not os.environ.get("QT_MEDIA_BACKEND"):
            os.environ["QT_MEDIA_BACKEND"] = "windows"
        try:
            from PySide6.QtCore import QUrl  # noqa: F401
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        except ImportError as exc:
            log.warning(
                "voice_tutorial: QtMultimedia unavailable (%s)", exc,
            )
            return False
        self._audio_output = QAudioOutput()
        self._audio_output.setVolume(1.0)
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_output)
        self._player.playbackStateChanged.connect(self._on_playback_state)
        self._player.errorOccurred.connect(self._on_error)
        return True

    def _on_playback_state(self, state) -> None:
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            label = {
                QMediaPlayer.StoppedState: "stopped",
                QMediaPlayer.PlayingState: "playing",
                QMediaPlayer.PausedState: "paused",
            }.get(state, "?")
        except Exception:
            label = "?"
        if self._on_state_change is not None:
            try:
                self._on_state_change(label)
            except Exception:
                pass

    def _on_error(self, err, msg: str = "") -> None:
        log.warning("voice_tutorial: player error %s: %s", err, msg)
        if self._on_state_change is not None:
            try:
                self._on_state_change(f"error: {msg or err}")
            except Exception:
                pass

    def play(self, path: Path) -> bool:
        if not self._ensure_player():
            return False
        try:
            from PySide6.QtCore import QUrl
            self._player.setSource(QUrl.fromLocalFile(str(path)))
            self._player.play()
            return True
        except Exception as exc:
            log.warning("voice_tutorial: play failed: %s", exc)
            return False

    def stop(self) -> None:
        if self._player is None:
            return
        try:
            self._player.stop()
        except Exception:
            pass

    def pause(self) -> None:
        """Pause playback. ``resume()`` continues from the same position;
        ``play(path)`` would restart from the beginning."""
        if self._player is None:
            return
        try:
            self._player.pause()
        except Exception:
            pass

    def resume(self) -> None:
        """Resume from the paused position. No-op if not currently paused
        (Qt's ``play()`` on a stopped player would restart from 0, which
        we explicitly avoid here so ``resume()`` is a safe one-way op).
        """
        if self._player is None:
            return
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            if self._player.playbackState() == QMediaPlayer.PausedState:
                self._player.play()
        except Exception:
            pass

    def is_playing(self) -> bool:
        if self._player is None:
            return False
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            return self._player.playbackState() == QMediaPlayer.PlayingState
        except Exception:
            return False

    def is_paused(self) -> bool:
        if self._player is None:
            return False
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            return self._player.playbackState() == QMediaPlayer.PausedState
        except Exception:
            return False
