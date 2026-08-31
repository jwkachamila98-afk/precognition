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
