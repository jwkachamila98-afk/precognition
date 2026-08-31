"""Voice guidance playback (tests/test_audio_playback.py).

The guidance voice crackled. The synthesised audio itself was clean - correct
rate, no header, no clipping - so the fault was in playback: the model returns
24 kHz, the speakers run natively at 48 kHz, and the microphone holds the same
physical device open at 16 kHz, all while the render loop saturates the CPU.
"""

import types

import numpy as np
import pytest

from src.audio.text_to_speech import GeminiSpeaker


def _speaker(device_rate):
    sp = GeminiSpeaker.__new__(GeminiSpeaker)
    sp._sd = types.SimpleNamespace(
        default=types.SimpleNamespace(device=(0, 1)),
        query_devices=lambda idx: {"default_samplerate": device_rate})
    return sp


def _tone(n, rate, hz=440.0, amp=20000):
    t = np.arange(n) / float(rate)
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


@pytest.mark.parametrize("device_rate", [48000, 44100, 22050])
def test_audio_is_converted_to_the_output_device_rate(device_rate):
    """Leaving CoreAudio to reconcile three different rates on a saturated CPU
    is what made playback crackle. Convert once, on a background thread."""
    src_rate = 24000
    audio = _tone(24000, src_rate)
    out, out_rate = _speaker(device_rate)._match_device_rate(audio, src_rate)

    assert out_rate == device_rate
    # Duration is what must survive; sample count follows from the rate.
    assert abs(len(out) / out_rate - len(audio) / src_rate) < 0.01
    assert out.dtype == np.int16
    assert np.abs(out).max() <= 32767, "resampling overshoot must not clip"


def test_no_conversion_when_the_rates_already_match():
    audio = _tone(1000, 48000)
    out, rate = _speaker(48000)._match_device_rate(audio, 48000)
    assert rate == 48000 and out is audio


def test_a_device_that_cannot_be_queried_is_left_alone():
    """Never fail to speak because the device could not be interrogated."""
    sp = GeminiSpeaker.__new__(GeminiSpeaker)
    sp._sd = types.SimpleNamespace(
        default=types.SimpleNamespace(device=(0, 1)),
        query_devices=lambda idx: (_ for _ in ()).throw(RuntimeError("no device")))
    audio = _tone(1000, 24000)
    out, rate = sp._match_device_rate(audio, 24000)
    assert rate == 24000 and out is audio


def test_the_waveform_survives_conversion():
    """A resampled tone must still be that tone, not noise - the failure mode
    being diagnosed here sounds exactly like a waveform turned to hash."""
    src_rate, hz = 24000, 440.0
    audio = _tone(24000, src_rate, hz=hz)
    out, out_rate = _speaker(48000)._match_device_rate(audio, src_rate)

    spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
    peak_hz = np.fft.rfftfreq(len(out), 1.0 / out_rate)[np.argmax(spectrum)]
    assert abs(peak_hz - hz) < 5.0, f"dominant tone moved to {peak_hz:.0f} Hz"


def _external_speaker(monkeypatch, player="/usr/bin/afplay"):
    import src.audio.text_to_speech as tts
    sp = GeminiSpeaker.__new__(GeminiSpeaker)
    sp._player = None
    monkeypatch.setattr(tts.shutil, "which", lambda name: player if name == "afplay" else None)
    return sp, tts


def test_guidance_plays_in_a_separate_process(monkeypatch, tmp_path):
    """Playing through sounddevice opened and closed a PortAudio output stream
    per utterance while the microphone held the same device open, and a teardown
    corrupted the heap:

        abort <- malloc_zone_error <- PaUtil_TerminateBufferProcessor <- CloseStream

    A SIGABRT with no traceback, which killed a live session. A separate player
    process cannot corrupt this one's heap.
    """
    import threading
    import wave as wavemod

    sp, tts = _external_speaker(monkeypatch)
    seen = {}

    class FakeProc:
        def __init__(self, argv, **kw):
            seen["argv"] = argv
            with wavemod.open(argv[1], "rb") as w:
                seen["rate"] = w.getframerate()
                seen["channels"] = w.getnchannels()
                seen["width"] = w.getsampwidth()
                seen["frames"] = w.getnframes()
            self._polls = 0

        def poll(self):
            self._polls += 1
            return None if self._polls < 3 else 0

        def terminate(self):
            seen["terminated"] = True

    monkeypatch.setattr(tts.subprocess, "Popen", FakeProc)

    pcm = _tone(2400, 24000).tobytes()
    assert sp._play_externally(pcm, 24000, threading.Event()) is True
    assert seen["argv"][0].endswith("afplay")
    assert (seen["rate"], seen["channels"], seen["width"]) == (24000, 1, 2)
    assert seen["frames"] == 2400, "the whole clip must reach the file"
    assert sp._player is None, "the handle must be cleared once playback ends"


def test_falls_back_to_sounddevice_when_no_player_exists(monkeypatch):
    import threading
    sp, _ = _external_speaker(monkeypatch, player=None)
    assert sp._play_externally(_tone(100, 24000).tobytes(), 24000, threading.Event()) is False


def test_a_stop_request_kills_the_player(monkeypatch):
    import threading
    sp, tts = _external_speaker(monkeypatch)
    killed = {}

    class NeverEnds:
        def __init__(self, argv, **kw):
            pass

        def poll(self):
            return None

        def terminate(self):
            killed["yes"] = True

    monkeypatch.setattr(tts.subprocess, "Popen", NeverEnds)
    ev = threading.Event()
    ev.set()
    assert sp._play_externally(_tone(100, 24000).tobytes(), 24000, ev) is True
    assert killed.get("yes"), "an interrupted clip must stop playing"
