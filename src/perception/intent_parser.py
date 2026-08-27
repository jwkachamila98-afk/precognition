"""Structured LLM Intent Parser and semantic schema grounding."""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """Structured semantic intent representation parsed from voice or text input."""
    target_object: str
    spatial_attributes: List[str] = field(default_factory=list)
    action_type: str = "reach_and_grasp"
    affordance_hotspot: str = "body"
    confidence_score: float = 0.95
    raw_transcript: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_object": self.target_object,
            "spatial_attributes": list(self.spatial_attributes),
            "action_type": self.action_type,
            "affordance_hotspot": self.affordance_hotspot,
            "confidence_score": float(self.confidence_score),
            "raw_transcript": self.raw_transcript
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedIntent":
        return cls(
            target_object=data.get("target_object", "none"),
            spatial_attributes=list(data.get("spatial_attributes", [])),
            action_type=data.get("action_type", "reach_and_grasp"),
            affordance_hotspot=data.get("affordance_hotspot", "body"),
            confidence_score=float(data.get("confidence_score", 0.95)),
            raw_transcript=data.get("raw_transcript", "")
        )

    @property
    def is_active(self) -> bool:
        """Return True if an actionable target object is specified."""
        return bool(self.target_object and self.target_object.lower() not in ("none", "idle", "clear", ""))


class IntentParserABC(ABC):
    """Abstract Base Class for semantic natural language intent parsing."""

    @abstractmethod
    def parse_intent(self, raw_transcript: str) -> ParsedIntent:
        """
        Parse unconstrained natural language transcript into structured schema.

        Args:
            raw_transcript: Speech-to-text transcript string (e.g. 'grasp the red mug by its handle').

        Returns:
            ParsedIntent dataclass instance.
        """
        pass


class MockLLMIntentParser(IntentParserABC):
    """
    Lightweight rule-based semantic parser simulating an LLM reasoning engine.
    Executes in <0.1 ms on CPU without requiring GPU memory or external API keys.
    """

    KNOWN_OBJECTS = {
        "remote": "remote control",
        "remote control": "remote control",
        "controller": "remote control",
        "cup": "coffee cup",
        "mug": "coffee cup",
        "coffee": "coffee cup",
        "bottle": "water bottle",
        "water bottle": "water bottle",
        "pen": "stylus pen",
        "stylus": "stylus pen",
        "box": "cardboard box",
        "phone": "smartphone",
        "smartphone": "smartphone",
        "apple": "apple",
    }

    KNOWN_ATTRIBUTES = [
        "red", "blue", "green", "black", "white", "yellow", "silver",
        "tall", "small", "large", "tiny", "heavy",
        "left", "right", "center", "table", "keyboard", "near", "front"
    ]

    KNOWN_ACTIONS = {
        "pick up": "reach_and_grasp",
        "pick": "reach_and_grasp",
        "grasp": "reach_and_grasp",
        "grab": "reach_and_grasp",
        "take": "reach_and_grasp",
        "lift": "lift_manipulate",
        "pinch": "precision_pinch",
        "hand over": "handover",
        "clear": "clear_standby",
        "stop": "clear_standby",
        "idle": "clear_standby"
    }

    KNOWN_AFFORDANCES = {
        "handle": "handle",
        "body": "body",
        "sides": "sides",
        "neck": "neck",
        "cap": "cap",
        "top": "top",
        "buttons": "buttons",
        "rim": "rim"
    }

    def parse_intent(self, raw_transcript: str) -> ParsedIntent:
        if not raw_transcript or raw_transcript.strip().lower() in ("idle", "none", "clear", "standby", ""):
            return ParsedIntent(
                target_object="none",
                spatial_attributes=[],
                action_type="clear_standby",
                affordance_hotspot="none",
                confidence_score=1.0,
                raw_transcript=raw_transcript
            )

        transcript_lower = raw_transcript.lower()

        # Handle clear/standby command explicitly
        if any(w in transcript_lower for w in ["clear", "standby", "stop", "reset"]):
            return ParsedIntent(
                target_object="none",
                spatial_attributes=[],
                action_type="clear_standby",
                affordance_hotspot="none",
                confidence_score=1.0,
                raw_transcript=raw_transcript
            )

        # 1. Target Object Extraction
        target_obj = "none"
        for key, canonical_name in self.KNOWN_OBJECTS.items():
            if re.search(rf"\b{re.escape(key)}\b", transcript_lower):
                target_obj = canonical_name
                break

        # Fallback noun matching (e.g. "pick up the stapler")
        if target_obj == "none":
            match = re.search(r"(?:pick\s+up|grasp|grab|take|get|reach\s+for|lift|hold)\s+(?:the\s+|a\s+|an\s+|this\s+)?([a-zA-Z_-]+)", transcript_lower)
            if match:
                extracted = match.group(1).strip()
                if extracted not in ("it", "object", "item", "something", "up", "the", "and", "target", "standby"):
                    target_obj = extracted

        # Final fallback: a short (1-2 word) bare noun phrase with no wrapping verb at
        # all (e.g. just saying "wine glass" into the mic) becomes the target itself,
        # rather than being silently dropped as "none". Capped at 2 words since every
        # COCO-80 class name/alias is at most two words - longer phrases are far more
        # likely to be noise/garbled transcription than a real object name.
        if target_obj == "none":
            stripped = re.sub(r"[^\w\s]", "", transcript_lower).strip()
            words = stripped.split()
            filler_words = {"please", "can", "you", "give", "me", "to", "for", "a", "an", "the"}
            if 1 <= len(words) <= 2 and not (set(words) & filler_words):
                target_obj = stripped

        # 2. Action Type Extraction
        action_type = "reach_and_grasp"
        for act_key, canonical_act in self.KNOWN_ACTIONS.items():
            if act_key in transcript_lower:
                action_type = canonical_act
                break

        # 3. Spatial & Visual Attributes
        attributes = []
        for attr in self.KNOWN_ATTRIBUTES:
            if re.search(rf"\b{attr}\b", transcript_lower):
                attributes.append(attr)

        # 4. Affordance Hotspot Extraction
        affordance = "body"
        for aff_key, canonical_aff in self.KNOWN_AFFORDANCES.items():
            if aff_key in transcript_lower:
                affordance = canonical_aff
                break
        if "cup" in target_obj and "handle" not in transcript_lower:
            affordance = "body" # Default cup grasp
        elif "remote" in target_obj:
            affordance = "sides"

        confidence = 0.96 if target_obj != "none" else 0.50

        return ParsedIntent(
            target_object=target_obj,
            spatial_attributes=attributes,
            action_type=action_type,
            affordance_hotspot=affordance,
            confidence_score=confidence,
            raw_transcript=raw_transcript
        )


class StructuredLLMIntentParser(IntentParserABC):
    """
    Zero-shot LLM intent parser connecting to local Ollama (e.g. Llama-3-8B / Qwen-2.5)
    or OpenAI / vLLM HTTP endpoints. Automatically falls back to MockLLMIntentParser
    on network or formatting errors.
    """

    SYSTEM_PROMPT = """You are a robotic vision & visuomotor manipulation intent parser.
Given a user voice instruction, output a strict JSON object with fields:
- "target_object": string (e.g. "coffee cup", "remote control", "water bottle", or "none")
- "spatial_attributes": list of strings (e.g. ["red", "on table"])
- "action_type": string (e.g. "reach_and_grasp", "precision_pinch", "lift_manipulate", "clear_standby")
- "affordance_hotspot": string (e.g. "handle", "body", "cap", "sides", "buttons")
- "confidence_score": float between 0.0 and 1.0
Output ONLY valid JSON without markdown wrapping."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:11434/api/generate", # Default local Ollama
        model_name: str = "llama3:8b",
        timeout: float = 1.5
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout = timeout
        self._fallback_parser = MockLLMIntentParser()

    def parse_intent(self, raw_transcript: str) -> ParsedIntent:
        if not raw_transcript or raw_transcript.strip().lower() in ("idle", "none", "clear", ""):
            return self._fallback_parser.parse_intent(raw_transcript)

        payload = {
            "model": self.model_name,
            "prompt": f"{self.SYSTEM_PROMPT}\n\nUser instruction: \"{raw_transcript}\"\nJSON Output:",
            "stream": False,
            "format": "json"
        }

        try:
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "{}")
                data = json.loads(response_text)
                return ParsedIntent(
                    target_object=data.get("target_object", "none"),
                    spatial_attributes=list(data.get("spatial_attributes", [])),
                    action_type=data.get("action_type", "reach_and_grasp"),
                    affordance_hotspot=data.get("affordance_hotspot", "body"),
                    confidence_score=float(data.get("confidence_score", 0.95)),
                    raw_transcript=raw_transcript
                )
        except Exception:
            # Fall back instantaneously to fast rule-based engine
            return self._fallback_parser.parse_intent(raw_transcript)
