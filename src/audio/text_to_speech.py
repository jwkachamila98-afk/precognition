"""Voice Guidance Output (TTS) Interfaces.

Gives the workflow state machine a voice: each phase transition in the
Foresee-then-Execute lifecycle can announce what the system is doing or
what the user should do next.
"""

import base64
import json
import logging
import re
import shutil
import ssl
import subprocess
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechSynthesizerABC(ABC):
    """Abstract Base Class for spoken voice guidance output."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak (or announce) the given text, interrupting any utterance in progress."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Immediately silence any utterance in progress."""
        pass

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """Return True if an utterance is currently in progress."""
        pass


class MockSpeaker(SpeechSynthesizerABC):
    """Silent speaker for headless/CI environments. Logs the utterance instead of playing audio."""

    def __init__(self) -> None:
        self.last_utterance: str = ""
        self._speaking_until = 0.0

    def speak(self, text: str) -> None:
        self.last_utterance = text
        # Approximate a natural speaking duration so `is_speaking` behaves plausibly.
        self._speaking_until = time.time() + max(0.6, 0.35 * len(text.split()))
        logger.info(f"[Voice Guidance] {text}")

    def stop(self) -> None:
        self._speaking_until = 0.0

    @property
    def is_speaking(self) -> bool:
        return time.time() < self._speaking_until


class SystemSpeaker(SpeechSynthesizerABC):
    """
    Zero-dependency TTS using the macOS built-in 'say' command.
    Falls back to MockSpeaker (silent logging) on any other platform or if 'say' is unavailable.
    """

    def __init__(self, rate_wpm: int = 195, voice: Optional[str] = None) -> None:
        self.rate_wpm = rate_wpm
        self.voice = voice
        self._proc: Optional[subprocess.Popen] = None
        self._available = shutil.which("say") is not None
        self._fallback = MockSpeaker()

        if not self._available:
            logger.info("SystemSpeaker: 'say' binary not found on this platform. Using silent fallback logging.")

    def speak(self, text: str) -> None:
        if not text:
            return

        if not self._available:
            self._fallback.speak(text)
            return

        # Interrupt any in-progress utterance so guidance never overlaps/queues stale phases.
        self.stop()

        cmd = ["say", "-r", str(self.rate_wpm)]
        if self.voice:
            cmd += ["-v", self.voice]
        cmd.append(text)

        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"[Voice Guidance] {text}")
        except Exception as e:
            logger.warning(f"SystemSpeaker failed to spawn 'say' ({e}). Falling back to silent logging.")
            self._fallback.speak(text)

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None

    @property
    def is_speaking(self) -> bool:
        if not self._available:
            return self._fallback.is_speaking
        return self._proc is not None and self._proc.poll() is None


class GeminiSpeaker(SpeechSynthesizerABC):
    """
    TTS via Gemini's native audio output (gemini-3.1-flash-tts-preview by default),
    played back locally through sounddevice. Synthesis + playback run on a background
    thread so a slow network call never blocks the render loop. Falls back to
    SystemSpeaker ('say') on any failure - missing key, network error, malformed
    response - so voice guidance never goes silent.
    """

    _ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-tts-preview",
        voice: str = "Kore",
        timeout: float = 8.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.timeout = timeout
        self._ssl_ctx = ssl.create_default_context()
        try:
            import certifi
            self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass

        self._fallback = SystemSpeaker()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._speaking = False

        try:
            import sounddevice as sd
            self._sd = sd
        except Exception as e:
            logger.warning(f"GeminiSpeaker: sounddevice unavailable ({e}); will use local 'say' fallback only.")
            self._sd = None

    def speak(self, text: str) -> None:
        if not text:
            return
        self.stop()
        if self._sd is None:
            self._fallback.speak(text)
            return

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._speak_worker, args=(text, self._stop_event), daemon=True)
        self._thread.start()

    def _speak_worker(self, text: str, stop_event: threading.Event) -> None:
        self._speaking = True
        try:
            pcm, sample_rate = self._synthesize(text)
            if stop_event.is_set():
                return
            import numpy as np
            audio = np.frombuffer(pcm, dtype=np.int16)
            audio, sample_rate = self._match_device_rate(audio, sample_rate)
            clip_duration_sec = len(audio) / float(sample_rate)
            # Hard ceiling well above the clip's own duration, in case CoreAudio ever
            # leaves a stream reporting active=True indefinitely (observed once during
            # testing, likely a device warm-up glitch) - playback must never hang
            # voice guidance forever.
            deadline = time.time() + clip_duration_sec + 3.0

            # A generous output buffer. The render loop saturates the CPU
            # (~35 ms in the window blit alone), and with the default low-latency
            # buffer the playback callback is starved often enough to crackle -
            # which is what "static" in the voice guidance actually was. Latency
            # is irrelevant here: this is a pre-rendered clip, not a live monitor.
            self._sd.play(audio, samplerate=sample_rate, latency="high")
            while self._sd.get_stream().active and not stop_event.is_set():
                if time.time() > deadline:
                    logger.warning("GeminiSpeaker: playback exceeded expected duration; forcing stop.")
                    self._sd.stop()
                    break
                time.sleep(0.05)
            if stop_event.is_set():
                self._sd.stop()
        except Exception as e:
            logger.warning(f"GeminiSpeaker: TTS failed ({e}); falling back to local 'say'.")
            if not stop_event.is_set():
                self._fallback.speak(text)
        finally:
            self._speaking = False

    def _match_device_rate(self, audio, sample_rate: int):
        """Resample to the output device's own rate before playing.

        The built-in speakers run natively at 48 kHz while the model returns
        24 kHz and the microphone stream holds the same physical device open at
        16 kHz. Leaving CoreAudio to reconcile all three, on a machine whose CPU
        is already saturated by the render loop, is what makes playback crackle.
        Converting once, here, costs a few milliseconds on a background thread.
        """
        import numpy as np
        try:
            device_rate = int(self._sd.query_devices(
                self._sd.default.device[1])["default_samplerate"])
        except Exception:
            return audio, sample_rate
        if device_rate <= 0 or device_rate == sample_rate:
            return audio, sample_rate
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(device_rate, sample_rate)
            converted = resample_poly(audio.astype(np.float32),
                                      device_rate // g, sample_rate // g)
        except Exception:
            # Linear interpolation is a fair fallback for speech.
            n = int(round(len(audio) * device_rate / float(sample_rate)))
            if n <= 1:
                return audio, sample_rate
            converted = np.interp(np.linspace(0, len(audio) - 1, n),
                                  np.arange(len(audio)), audio.astype(np.float32))
        peak = float(np.abs(converted).max()) if len(converted) else 0.0
        if peak > 32767.0:                       # resampling can overshoot slightly
            converted *= 32767.0 / peak
        return converted.astype(np.int16), device_rate

    def _synthesize(self, text: str) -> tuple:
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}},
            },
        }
        url = f"{self._ENDPOINT_TEMPLATE.format(model=self.model)}?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        part = result["candidates"][0]["content"]["parts"][0]["inlineData"]
        pcm = base64.b64decode(part["data"])
        mime = part.get("mimeType", "")
        rate_match = re.search(r"rate=(\d+)", mime)
        sample_rate = int(rate_match.group(1)) if rate_match else 24000
        return pcm, sample_rate

    def stop(self) -> None:
        self._stop_event.set()
        if self._sd is not None:
            try:
                self._sd.stop()
            except Exception:
                pass
        self._fallback.stop()

    @property
    def is_speaking(self) -> bool:
        return self._speaking or self._fallback.is_speaking
