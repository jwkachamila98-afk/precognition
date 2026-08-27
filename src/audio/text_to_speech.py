"""Voice Guidance Output (TTS) Interfaces.

Gives the workflow state machine a voice: each phase transition in the
Foresee-then-Execute lifecycle can announce what the system is doing or
what the user should do next.
"""

import logging
import shutil
import subprocess
import time
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
