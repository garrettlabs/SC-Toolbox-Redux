"""Tiny Qt window to audition the CABAL TTS voice.

Run::

    python -m voice.tester

Type text, hit Play. Toggle "Dry (bypass effects)" to compare the
raw Piper voice against the processed CABAL timbre. First Play
downloads the voice model (~60 MB) — subsequent plays are instant.
"""

from __future__ import annotations

import logging
import sys
import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
    QPushButton, QCheckBox, QLabel, QComboBox,
)

from .cabal_tts import synthesize, VoiceConfig, EffectsConfig


_PRESETS = [
    "Scan complete. Aluminum ore detected. "
    "Mass five thousand six hundred seven kilograms. Rock is breakable.",

    "Warning. Instability rising. Recommend reducing laser throttle.",

    "Target resistance exceeds current power budget. "
    "Cannot break. Requesting fleet assistance.",

    "Refinery order complete. Twelve point four S C U of quantanium, ready for pickup.",

    "I am CABAL. Designation: Computer Assisted Biologically Augmented Lifeform.",
]


class VoiceTester(QWidget):
    _audio_ready = Signal(object, int)
    _status = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CABAL Voice Tester")
        self.setMinimumSize(620, 360)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)

        root.addWidget(QLabel("<b>CABAL Voice Tester</b>"))

        # ── Preset picker ──
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.presets = QComboBox()
        self.presets.addItems([f"{i+1}. {p[:60]}…" for i, p in enumerate(_PRESETS)])
        self.presets.currentIndexChanged.connect(
            lambda i: self.text.setPlainText(_PRESETS[i]),
        )
        preset_row.addWidget(self.presets, 1)
        root.addLayout(preset_row)

        # ── Text box ──
        self.text = QPlainTextEdit(_PRESETS[0])
        self.text.setPlaceholderText("Type what you want spoken…")
        root.addWidget(self.text, 1)

        # ── Options ──
        opts = QHBoxLayout()
        self.dry = QCheckBox("Dry (bypass effects)")
        opts.addWidget(self.dry)

        opts.addWidget(QLabel("Voice:"))
        self.voice_pick = QComboBox()
        self.voice_pick.addItems([
            "en_GB-alan-medium",
            "en_US-lessac-medium",
            "en_US-ryan-medium",
        ])
        opts.addWidget(self.voice_pick)
        opts.addStretch(1)
        root.addLayout(opts)

        # ── Buttons ──
        btns = QHBoxLayout()
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setDefault(True)
        self.play_btn.clicked.connect(self._on_play)
        btns.addWidget(self.play_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)
        root.addLayout(btns)

        # ── Status ──
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setStyleSheet("color: #888;")
        root.addWidget(self.status_lbl)

        self._audio_ready.connect(self._play_audio)
        self._status.connect(self.status_lbl.setText)

    def _on_play(self):
        text = self.text.toPlainText().strip()
        if not text:
            return
        self.play_btn.setEnabled(False)
        self._status.emit("Synthesizing…")

        vcfg = VoiceConfig(model=self.voice_pick.currentText())
        fcfg = EffectsConfig()
        if self.dry.isChecked():
            for k in list(fcfg.__dict__):
                if k.endswith("_enabled"):
                    setattr(fcfg, k, False)

        def _worker():
            try:
                audio, sr = synthesize(text, voice=vcfg, effects=fcfg)
                self._audio_ready.emit(audio, sr)
            except Exception as exc:
                self._status.emit(f"Error: {exc}")
                logging.exception("tester synth failed")
            finally:
                # Re-enable button on the GUI thread via status signal
                # (cheap trick — any slot fires on main thread).
                self._status.emit(self.status_lbl.text())

        threading.Thread(target=_worker, daemon=True).start()

    def _play_audio(self, audio, sr: int):
        import sounddevice as sd
        try:
            sd.stop()
            sd.play(audio, sr)
            dur_ms = int(1000 * len(audio) / sr)
            self._status.emit(f"Playing — {dur_ms} ms @ {sr} Hz")
        except Exception as exc:
            self._status.emit(f"Playback error: {exc}")
        finally:
            self.play_btn.setEnabled(True)

    def _on_stop(self):
        try:
            import sounddevice as sd
            sd.stop()
            self._status.emit("Stopped.")
        except Exception:
            pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = QApplication.instance() or QApplication(sys.argv)
    w = VoiceTester()
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
