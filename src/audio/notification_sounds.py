"""Calming notification tones (src/audio/notification_sounds.py).

Spoken guidance was replaced by short tones. Speech was the wrong instrument
here: it arrives a second or two after the moment it describes, it talks over
the person who is trying to concentrate on a reach, it burned Gemini quota that
transcription and detection need more, and every utterance opened an audio
stream - which is what eventually corrupted the heap and killed a session.

A tone is immediate, costs nothing, and says "something changed" without
demanding attention. The palette is deliberately gentle: sine partials in
consonant intervals, a soft attack so nothing clicks, a long decay, and a low
enough level to sit under a conversation.

Clips are synthesised once at startup and played by the OS player, out of
process, for the same reason the speech was.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_RATE = 44100

# (frequency Hz, start seconds, duration seconds, gain)
_Voicing = List[Tuple[float, float, float, float]]

# Notes around the middle of the piano, where tones read as warm rather than
# shrill. Intervals are consonant - fifths, fourths, major thirds - so a cue
# never sounds like an alarm.
_C5, _D5, _E5, _G5, _A5, _C6 = 523.25, 587.33, 659.25, 783.99, 880.00, 1046.50
_G4, _A4 = 392.00, 440.00

CUES: Dict[str, _Voicing] = {
    # Something is ready for you: a rising fourth, unhurried.
    "ready":     [(_G4, 0.00, 0.45, 0.55), (_C5, 0.11, 0.55, 0.55)],
    # Listening: one soft note, so it never competes with the speaking voice.
    "listening": [(_A4, 0.00, 0.30, 0.42)],
    # Heard you: the same note answered a fifth above.
    "heard":     [(_A4, 0.00, 0.26, 0.42), (_E5, 0.09, 0.42, 0.42)],
    # Go: two quick ascending notes, the most urgent thing in the set.
    "go":        [(_C5, 0.00, 0.22, 0.60), (_G5, 0.10, 0.40, 0.60)],
    # Finished and scored: a settled major third, falling to rest.
    "complete":  [(_G5, 0.00, 0.30, 0.50), (_E5, 0.12, 0.55, 0.50)],
    # Learned something: a bright, quiet arpeggio.
    "improved":  [(_C5, 0.00, 0.22, 0.40), (_E5, 0.09, 0.24, 0.40),
                  (_G5, 0.18, 0.45, 0.40)],
    # Attention: low and soft. Deliberately NOT a buzzer.
    "attention": [(_D5, 0.00, 0.26, 0.45), (_G4, 0.13, 0.50, 0.45)],
}


def _render(voicing: _Voicing, volume: float) -> np.ndarray:
    """Additive synthesis with a soft attack and an exponential decay."""
    total = max(start + dur for _, start, dur, _ in voicing) + 0.05
    out = np.zeros(int(total * _RATE), dtype=np.float64)
    for freq, start, dur, gain in voicing:
        n = int(dur * _RATE)
        t = np.arange(n) / _RATE
        # A touch of octave above for warmth; too much and it turns glassy.
        wave_ = np.sin(2 * math.pi * freq * t) + 0.18 * np.sin(4 * math.pi * freq * t)
        # 12 ms attack removes the click of a hard start; the decay is long
        # enough that the tone fades rather than stops.
        attack = np.clip(t / 0.012, 0.0, 1.0)
        decay = np.exp(-t / (dur * 0.42))
        seg = wave_ * attack * decay * gain
        at = int(start * _RATE)
        out[at:at + n] += seg[: len(out) - at]
    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * float(np.clip(volume, 0.0, 1.0))
    return (out * 32767.0).astype(np.int16)


class NotificationSounds:
    """Short calming cues, rendered once and played out of process."""

    def __init__(self, enabled: bool = True, volume: float = 0.32) -> None:
        self.enabled = bool(enabled)
        self.volume = float(volume)
        self._paths: Dict[str, str] = {}
        self._dir: Optional[str] = None
        self._player = shutil.which("afplay") or shutil.which("aplay")
        self._procs: List[subprocess.Popen] = []
        if self._player:
            self._render_all()
        else:
            logger.info("NotificationSounds: no system audio player; cues disabled.")

    def _render_all(self) -> None:
        try:
            self._dir = tempfile.mkdtemp(prefix="precog-cues-")
            for name, voicing in CUES.items():
                samples = _render(voicing, self.volume)
                path = os.path.join(self._dir, f"{name}.wav")
                with wave.open(path, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(_RATE)
                    wav.writeframes(samples.tobytes())
                self._paths[name] = path
        except Exception as exc:
            logger.warning(f"NotificationSounds: could not prepare cues ({exc}).")
            self._paths = {}

    def play(self, cue: str) -> bool:
        """Sound a cue. Never blocks, never raises, never queues up a backlog."""
        if not self.enabled or not self._player:
            return False
        path = self._paths.get(cue)
        if not path:
            return False
        self._reap()
        # A cue that is already sounding is not worth stacking a second copy on.
        if len(self._procs) >= 3:
            return False
        try:
            self._procs.append(subprocess.Popen(
                [self._player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            return True
        except Exception:
            return False

    def _reap(self) -> None:
        self._procs = [p for p in self._procs if p.poll() is None]

    def stop(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        self._procs = []

    def close(self) -> None:
        self.stop()
        if self._dir and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
