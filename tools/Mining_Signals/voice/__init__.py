"""CABAL-style offline text-to-speech.

Usage::

    from voice import speak
    speak("Scan complete. Aluminum ore detected.")

See :mod:`voice.cabal_tts` for the full API.
"""

from .cabal_tts import speak, synthesize, save_wav, VoiceConfig, EffectsConfig

__all__ = ["speak", "synthesize", "save_wav", "VoiceConfig", "EffectsConfig"]
