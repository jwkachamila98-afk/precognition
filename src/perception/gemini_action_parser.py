"""Reading an arbitrary spoken action (src/perception/gemini_action_parser.py).

The rule table in `action_schema` covers the phrasings people reach for first.
It cannot cover what someone actually says - "tip the last bit out of that",
"nudge it towards me", "check if the lid is loose" - and a verb list that has to
grow every time a demo meets a new sentence is not a design.

So an arbitrary utterance is handed to a language model, which fills in the same
schema the rules produce. The schema is what makes this safe: the model is not
inventing motion, only describing it along axes the trajectory generator already
knows how to execute, and every field it returns is clamped on the way in.

Results are cached per utterance and requested off the render loop, because the
call costs a second or more.
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import urllib.request
from typing import Dict, Optional

import certifi

from src.perception.action_schema import (APPROACHES, CONTACTS, FOLLOW_THROUGHS,
                                          ActionPlan, plan_from_text)

logger = logging.getLogger(__name__)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_PROMPT = (
    "A person said: \"{utterance}\"\n\n"
    "Describe the hand motion they intend, as JSON with exactly these keys:\n"
    '  "verb": the action word they used, 1-2 words\n'
    '  "approach": one of {approaches} - where the hand arrives from\n'
    '  "contact": one of {contacts} - what the hand does on arrival\n'
    '  "grip": 0.0 to 1.0, how closed the hand ends up\n'
    '  "follow_through": one of {follows} - what happens after contact\n'
    '  "travel_m": metres the object or hand moves after contact, 0.0 to 0.45\n'
    '  "tilt_deg": wrist rotation after contact in degrees, -120 to 120\n\n'
    "Guidance: pouring or drinking tilts (40-70). Opening or unscrewing twists "
    "(60-90). Pushing and sliding use a side approach and little grip. Pointing "
    "makes no contact. Handing something over travels toward the user. "
    "Respond with ONLY the JSON object, no markdown fencing."
)


class GeminiActionParser:
    """Fills the action schema for any utterance, with the rules as a floor."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash",
                 timeout: float = 12.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._cache: Dict[str, ActionPlan] = {}
        self._lock = threading.Lock()
        self._inflight: set = set()

    def cached(self, utterance: str) -> Optional[ActionPlan]:
        """A previously resolved plan, without making a request."""
        with self._lock:
            return self._cache.get((utterance or "").strip().lower())

    def plan(self, utterance: str) -> ActionPlan:
        """Blocking. The model's reading, or the rules if it cannot be reached."""
        key = (utterance or "").strip().lower()
        if not key:
            return ActionPlan(source="default")
        cached = self.cached(key)
        if cached is not None:
            return cached

        fallback = plan_from_text(utterance)
        try:
            parsed = self._request(utterance)
        except Exception as exc:
            logger.info(f"GeminiActionParser: falling back to rules for "
                        f"'{utterance[:40]}' ({type(exc).__name__}).")
            parsed = None

        result = parsed if parsed is not None else fallback
        with self._lock:
            self._cache[key] = result
        return result

    def plan_async(self, utterance: str) -> ActionPlan:
        """Non-blocking: the rules now, the model's reading once it arrives.

        The render loop must never wait on a network round trip, and an action
        that sharpens a second after it is spoken is better than a stalled frame.
        """
        key = (utterance or "").strip().lower()
        if not key:
            return ActionPlan(source="default")
        cached = self.cached(key)
        if cached is not None:
            return cached
        with self._lock:
            if key not in self._inflight:
                self._inflight.add(key)
                threading.Thread(target=self._resolve_worker, args=(utterance, key),
                                 daemon=True, name="gemini-action").start()
        return plan_from_text(utterance)

    def _resolve_worker(self, utterance: str, key: str) -> None:
        try:
            self.plan(utterance)
        finally:
            with self._lock:
                self._inflight.discard(key)

    def _request(self, utterance: str) -> Optional[ActionPlan]:
        prompt = _PROMPT.format(
            utterance=utterance,
            approaches=list(APPROACHES), contacts=list(CONTACTS),
            follows=list(FOLLOW_THROUGHS))
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        url = f"{_ENDPOINT.format(model=self.model)}?key={self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        candidate = result["candidates"][0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise ValueError("response truncated")
        data = json.loads(candidate["content"]["parts"][0]["text"])
        data["source"] = "gemini"
        data.setdefault("confidence", 0.9)
        plan = ActionPlan.from_dict(data)
        logger.info(f"GeminiActionParser: '{utterance[:48]}' -> {plan.summary} "
                    f"(grip {plan.grip:.2f}, travel {plan.travel_m*100:.0f} cm, "
                    f"tilt {plan.tilt_deg:.0f} deg)")
        return plan
