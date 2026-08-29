"""Continuous Audio Ingestion and Speech-to-Text (STT) Interfaces."""

import base64
import collections
import io
import json
import logging
import ssl
import threading
import time
import urllib.request
import wave
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AudioTranscriberABC(ABC):
    """Abstract Base Class for continuous audio capture and speech-to-text transcription."""

    @property
    @abstractmethod
    def is_listening(self) -> bool:
        """Return True if audio capture stream is currently active."""
        pass

    @abstractmethod
    def start_listening(self) -> None:
        """Begin audio capture from microphone or virtual stream."""
        pass

    @abstractmethod
    def stop_listening(self) -> str:
        """Stop audio capture, finalize transcription, and return recognized text string."""
        pass

    @abstractmethod
    def transcribe_stream(self, audio_chunk: bytes) -> Optional[str]:
        """Process an incoming stream audio chunk and return intermediate transcription if available."""
        pass

    @abstractmethod
    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an existing audio file (.wav, .mp3, .flac)."""
        pass


class MockTranscriber(AudioTranscriberABC):
    """
    Lightweight CPU mock transcriber for local development without physical microphone/audio drivers.
    Cycles through or triggers realistic robotic manipulation voice instructions.
    """

    PRESET_VOICE_COMMANDS = [
        "foresee me picking this remote control",
        "grasp the red coffee cup by the handle",
        "pick up the tall water bottle on the right",
        "grab the stylus pen near the keyboard",
        "lift the small cardboard box from the table",
        "clear target and return to standby"
    ]

    def __init__(self, preset_commands: Optional[List[str]] = None) -> None:
        self.commands = preset_commands or list(self.PRESET_VOICE_COMMANDS)
        self._cmd_idx = 0
        self._listening = False
        self._start_time = 0.0

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start_listening(self) -> None:
        self._listening = True
        self._start_time = time.time()
        logger.info("Mock Voice Ingestion: [LISTENING... Speak your intent prompt]")

    def stop_listening(self) -> str:
        self._listening = False
        # Retrieve command and cycle
        transcript = self.commands[self._cmd_idx]
        self._cmd_idx = (self._cmd_idx + 1) % len(self.commands)
        logger.info(f"Mock Voice Ingestion: [TRANSCRIBED] -> '{transcript}'")
        return transcript

    def transcribe_stream(self, audio_chunk: bytes) -> Optional[str]:
        if self._listening and (time.time() - self._start_time > 1.5):
            return self.commands[self._cmd_idx]
        return None

    def transcribe_file(self, audio_path: str) -> str:
        return self.commands[0]


class WhisperTranscriber(AudioTranscriberABC):
    """
    Real-time Whisper STT transcriber utilizing faster-whisper / Silero VAD.
    Runs locally on CPU ('tiny.en' or 'base.en' quantized int8) or GPU with minimal latency.
    Falls back gracefully to MockTranscriber if audio libraries/models are unavailable.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self._listening = False
        self._model = None
        self._audio_buffer: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._fallback_mock = MockTranscriber()

        self._init_model()

    def _init_model(self) -> None:
        """Lazy load faster-whisper model if installed."""
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model '{self.model_size}' on {self.device} ({self.compute_type})...")
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            logger.info("faster-whisper model loaded successfully.")
        except ImportError:
            logger.info("faster-whisper is not installed. WhisperTranscriber will use synthetic voice engine fallback.")
            self._model = None
        except Exception as e:
            logger.warning(f"Could not initialize WhisperModel ({e}). Using synthetic voice engine fallback.")
            self._model = None

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start_listening(self) -> None:
        with self._lock:
            self._listening = True
            self._audio_buffer.clear()
        if self._model is None:
            self._fallback_mock.start_listening()
        else:
            logger.info("WhisperTranscriber: [LISTENING on microphone...]")

    def stop_listening(self) -> str:
        with self._lock:
            self._listening = False
            buffered = list(self._audio_buffer)
            self._audio_buffer.clear()

        if self._model is None or not buffered:
            return self._fallback_mock.stop_listening()

        try:
            # Concatenate audio chunks
            audio_data = np.concatenate(buffered, axis=0).astype(np.float32)
            segments, _ = self._model.transcribe(audio_data, language="en", beam_size=1)
            transcript = " ".join([seg.text.strip() for seg in segments])
            logger.info(f"WhisperTranscriber: [TRANSCRIBED] -> '{transcript}'")
            return transcript if transcript else "idle"
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return self._fallback_mock.stop_listening()

    def transcribe_stream(self, audio_chunk: bytes) -> Optional[str]:
        if not self._listening:
            return None
        # Convert raw PCM16 bytes to float32
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self._audio_buffer.append(audio_np)
        return None

    def transcribe_file(self, audio_path: str) -> str:
        if self._model is None:
            return self._fallback_mock.transcribe_file(audio_path)
        try:
            segments, _ = self._model.transcribe(audio_path, language="en")
            return " ".join([seg.text.strip() for seg in segments])
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return self._fallback_mock.transcribe_file(audio_path)


class GeminiTranscriber(AudioTranscriberABC):
    """
    Push-to-talk transcription via Gemini's native audio understanding. Buffers raw
    PCM16 audio the same way WhisperTranscriber does (fed by the same sounddevice
    microphone callback in local_client.py - no changes needed there), then sends the
    full clip to Gemini once listening stops. Falls back to MockTranscriber on any
    failure (missing key, network error, malformed response) so push-to-talk never
    silently does nothing.
    """

    _ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    _PROMPT = (
        "Transcribe this audio clip exactly as spoken. Respond with ONLY the "
        "transcribed text - no quotes, no punctuation commentary, no extra words."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        sample_rate: int = 16000,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._ssl_ctx = ssl.create_default_context()
        try:
            import certifi
            self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass

        self._listening = False
        self._audio_buffer: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._fallback = MockTranscriber()

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start_listening(self) -> None:
        with self._lock:
            self._listening = True
            self._audio_buffer.clear()
        logger.info("GeminiTranscriber: [LISTENING on microphone...]")

    def stop_listening(self) -> str:
        with self._lock:
            self._listening = False
            buffered = list(self._audio_buffer)
            self._audio_buffer.clear()

        if not buffered:
            # No audio captured at all - no microphone, or the device is muted.
            # This is the dev-without-a-mic path the presets exist for, and is
            # distinct from a request that failed on real recorded speech.
            logger.warning("GeminiTranscriber: no audio captured; using a preset utterance.")
            return self._fallback.stop_listening()

        try:
            audio_int16 = np.concatenate(buffered, axis=0)
            wav_bytes = self._to_wav_bytes(audio_int16, self.sample_rate)
            transcript = self._transcribe(wav_bytes)
            logger.info(f"GeminiTranscriber: [TRANSCRIBED] -> '{transcript}'")
            return transcript if transcript else "idle"
        except Exception as e:
            # Do NOT fall back to the mock here. Its presets are plausible
            # sentences ("foresee me picking this remote control"), so a failed
            # request used to hand back a fabricated utterance that was
            # indistinguishable from a real one in the logs and on screen - the
            # system would confidently chase an object the user never named, and
            # the intent embedding would encode a sentence they never said.
            # An empty transcript leaves the current target untouched instead.
            logger.error(
                f"GeminiTranscriber: transcription failed ({e}). Reporting no "
                f"transcript rather than guessing - say it again."
            )
            return ""

    def transcribe_stream(self, audio_chunk: bytes) -> Optional[str]:
        if not self._listening:
            return None
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        with self._lock:
            self._audio_buffer.append(audio_np)
        return None

    def transcribe_file(self, audio_path: str) -> str:
        try:
            with open(audio_path, "rb") as f:
                data = f.read()
            return self._transcribe(data)
        except Exception as e:
            logger.error(f"GeminiTranscriber: file transcription failed ({e})")
            return self._fallback.transcribe_file(audio_path)

    @staticmethod
    def _to_wav_bytes(audio_int16: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio_int16.astype(np.int16).tobytes())
        return buf.getvalue()

    def _transcribe(self, wav_bytes: bytes) -> str:
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        payload = {
            "contents": [{"parts": [
                {"text": self._PROMPT},
                {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
            ]}]
        }
        url = f"{self._ENDPOINT_TEMPLATE.format(model=self.model)}?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
