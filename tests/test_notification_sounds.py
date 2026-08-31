"""Notification cues (tests/test_notification_sounds.py).

Spoken guidance was replaced by short tones. Speech arrived a second or two
after the moment it described, talked over someone concentrating on a reach,
consumed Gemini quota that transcription and detection need more, and every
utterance opened an audio stream - which is what corrupted the heap and killed
a live session.
"""

import subprocess
import wave

import numpy as np
import pytest

from src.audio.notification_sounds import CUES, NotificationSounds, _render


@pytest.mark.parametrize("name", sorted(CUES))
def test_every_cue_renders_to_audible_finite_audio(name):
    samples = _render(CUES[name], 0.32)
    assert samples.dtype == np.int16
    assert np.all(np.isfinite(samples))
    assert 0.15 < len(samples) / 44100 < 1.2, "a cue should be brief"
    assert np.abs(samples).max() > 2000, "inaudible"
    assert np.abs(samples).max() <= 32767, "clipped"


@pytest.mark.parametrize("name", sorted(CUES))
def test_cues_start_softly_and_end_quietly(name):
    """A hard start clicks and a hard stop sounds like a fault. Calming means
    an attack you cannot hear beginning and a tail that fades."""
    samples = _render(CUES[name], 0.32).astype(np.float64)
    assert abs(samples[0]) < 40, "the attack clicks"
    tail = np.abs(samples[-int(0.02 * 44100):]).max()
    assert tail < np.abs(samples).max() * 0.12, "the tail is cut off rather than faded"


def test_volume_scales_without_clipping():
    quiet = np.abs(_render(CUES["ready"], 0.1)).max()
    loud = np.abs(_render(CUES["ready"], 0.8)).max()
    assert loud > quiet * 4
    assert loud <= 32767


def test_cues_are_consonant_not_alarming():
    """An alarm is a dissonance held at volume; this set should never be
    mistaken for one.

    Measured in semitones rather than against just-intonation ratios, because
    the notes are equal-tempered: a descending minor third is 0.8409, not the
    0.8333 of pure tuning, and comparing raw ratios flags it as dissonant.
    """
    import math

    # Unison, thirds, fourth, fifth, sixths, octave - the consonances.
    consonant_semitones = {0, 3, 4, 5, 7, 8, 9, 12}
    for name, voicing in CUES.items():
        freqs = [f for f, _, _, _ in voicing]
        for a, b in zip(freqs, freqs[1:]):
            semitones = abs(round(12 * math.log2(b / a)))
            assert semitones in consonant_semitones, \
                f"{name}: {a:.0f}->{b:.0f} is {semitones} semitones, a dissonance"


def test_playing_is_non_blocking_and_never_raises(monkeypatch, tmp_path):
    started = []

    class FakeProc:
        def __init__(self, argv, **kw):
            started.append(argv)

        def poll(self):
            return None

        def terminate(self):
            pass

    sounds = NotificationSounds(enabled=True)
    if not sounds._player:
        pytest.skip("no system audio player on this host")
    monkeypatch.setattr(subprocess, "Popen", FakeProc)

    assert sounds.play("ready") is True
    assert sounds.play("nonexistent-cue") is False, "an unknown cue must not raise"
    sounds.enabled = False
    assert sounds.play("ready") is False, "muting must actually mute"
    sounds.close()


def test_cues_never_pile_up(monkeypatch):
    """Rapid phase changes must not stack a dozen overlapping players."""
    class NeverEnds:
        def __init__(self, argv, **kw):
            pass

        def poll(self):
            return None

        def terminate(self):
            pass

    sounds = NotificationSounds(enabled=True)
    if not sounds._player:
        pytest.skip("no system audio player on this host")
    monkeypatch.setattr(subprocess, "Popen", NeverEnds)
    played = sum(sounds.play("go") for _ in range(20))
    assert played <= 3, f"{played} overlapping players"
    sounds.close()


def test_written_clips_are_valid_wav_files():
    sounds = NotificationSounds(enabled=True)
    if not sounds._paths:
        pytest.skip("no system audio player on this host")
    for name, path in sounds._paths.items():
        with wave.open(path, "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 44100
            assert wav.getnframes() > 4000, f"{name} is too short to hear"
    sounds.close()
