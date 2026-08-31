"""A parameterisation of manipulation actions (src/perception/action_schema.py).

The system used to recognise a fixed list of verbs and, having recognised one,
do nothing with it - `action_type` was parsed and then read by no one, so
"push the cup" and "pick up the cup" produced identical motion.

Enumerating verbs does not scale: people say "drink from this", "slide that out
of the way", "hand me the remote". So actions are not enumerated here. They are
DESCRIBED, along a handful of axes that together determine a reach:

    where the hand comes from      approach
    what it does on arrival        contact, grip
    what happens next             follow_through, travel, tilt

Any verb can be expressed this way, which is what lets an unseen phrase produce
sensible motion instead of falling back to a generic grasp. A language model
fills the fields in; the rule-based mapping below covers the common cases
offline and when the model is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Vocabulary of the schema itself. These ARE closed sets - not because actions
# are, but because motion has to be generated from something concrete.
# ---------------------------------------------------------------------------

APPROACHES = ("above", "side", "front")
CONTACTS = ("grasp", "pinch", "push", "touch", "none")
FOLLOW_THROUGHS = ("lift", "toward_user", "slide", "tilt", "hold", "retreat", "none")


@dataclass
class ActionPlan:
    """How a spoken action becomes a reach."""

    verb: str = "pick up"
    approach: str = "above"          # where the hand arrives from
    contact: str = "grasp"           # what it does on arrival
    grip: float = 0.85               # final closure, 0 open .. 1 closed
    follow_through: str = "lift"     # what happens after contact
    travel_m: float = 0.12           # how far the object moves afterwards
    tilt_deg: float = 0.0            # wrist rotation after contact
    confidence: float = 0.5
    source: str = "rules"            # "gemini" | "rules" | "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verb": self.verb, "approach": self.approach, "contact": self.contact,
            "grip": float(self.grip), "follow_through": self.follow_through,
            "travel_m": float(self.travel_m), "tilt_deg": float(self.tilt_deg),
            "confidence": float(self.confidence), "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ActionPlan":
        if not data:
            return cls()
        plan = cls(
            verb=str(data.get("verb", "pick up")),
            approach=str(data.get("approach", "above")),
            contact=str(data.get("contact", "grasp")),
            grip=float(data.get("grip", 0.85)),
            follow_through=str(data.get("follow_through", "lift")),
            travel_m=float(data.get("travel_m", 0.12)),
            tilt_deg=float(data.get("tilt_deg", 0.0)),
            confidence=float(data.get("confidence", 0.5)),
            source=str(data.get("source", "rules")),
        )
        return plan.validated()

    def validated(self) -> "ActionPlan":
        """Clamp to what the motion generator can actually produce.

        A language model will occasionally invent a field value or a travel
        distance of two metres; the trajectory generator must never be handed
        something it cannot express.
        """
        if self.approach not in APPROACHES:
            self.approach = "above"
        if self.contact not in CONTACTS:
            self.contact = "grasp"
        if self.follow_through not in FOLLOW_THROUGHS:
            self.follow_through = "lift"
        self.grip = float(min(max(self.grip, 0.0), 1.0))
        # A hand's reach, not a walk across the room.
        self.travel_m = float(min(max(self.travel_m, 0.0), 0.45))
        self.tilt_deg = float(min(max(self.tilt_deg, -120.0), 120.0))
        self.confidence = float(min(max(self.confidence, 0.0), 1.0))
        # Contact and follow-through have to agree: nothing can be carried by a
        # hand that never closed on it.
        if self.contact in ("none", "touch") and self.follow_through in ("lift", "toward_user", "tilt"):
            self.follow_through = "hold" if self.contact == "touch" else "retreat"
        if self.contact == "none":
            self.grip = min(self.grip, 0.25)
        return self

    @property
    def summary(self) -> str:
        """A short human-readable description, for the HUD and the logs."""
        parts = {
            "lift": "lift it", "toward_user": "bring it closer", "slide": "slide it",
            "tilt": "tilt it", "hold": "hold still", "retreat": "withdraw",
            "none": "stop there",
        }
        contact = {"grasp": "grasp", "pinch": "pinch", "push": "push",
                   "touch": "touch", "none": "reach toward"}[self.contact]
        return f"{contact} from {self.approach}, then {parts[self.follow_through]}"


# ---------------------------------------------------------------------------
# Offline mapping. Covers the phrasings people reach for first, and gives the
# language model something to fall back to rather than a generic grasp.
# ---------------------------------------------------------------------------

_RULES = (
    # (pattern, approach, contact, grip, follow_through, travel_m, tilt_deg)
    (r"\b(push|shove|nudge|slide|move)\b", "side", "push", 0.30, "slide", 0.15, 0.0),
    (r"\b(point|indicate|show me|gesture)\b", "front", "none", 0.15, "retreat", 0.0, 0.0),
    (r"\b(hand|pass|give|bring)\b", "side", "grasp", 0.85, "toward_user", 0.22, 0.0),
    (r"\b(drink|sip|pour|tip)\b", "side", "grasp", 0.80, "tilt", 0.10, 55.0),
    (r"\b(press|tap|poke|touch|click)\b", "above", "touch", 0.25, "hold", 0.0, 0.0),
    (r"\b(pinch|tweeze)\b", "above", "pinch", 0.95, "lift", 0.10, 0.0),
    (r"\b(put down|set down|place|drop|lower)\b", "above", "grasp", 0.85, "slide", 0.08, 0.0),
    (r"\b(open|turn|twist|rotate|unscrew)\b", "above", "grasp", 0.90, "tilt", 0.02, 75.0),
    (r"\b(pick|grab|take|lift|grasp|hold|get)\b", "above", "grasp", 0.85, "lift", 0.14, 0.0),
)


def plan_from_text(utterance: str) -> ActionPlan:
    """Best offline reading of a spoken action.

    Deliberately permissive: an unrecognised verb yields a plain grasp rather
    than nothing, because a sensible default beats refusing to move.
    """
    said = (utterance or "").lower().strip()
    for pattern, approach, contact, grip, follow, travel, tilt in _RULES:
        match = re.search(pattern, said)
        if match:
            return ActionPlan(
                verb=match.group(1), approach=approach, contact=contact, grip=grip,
                follow_through=follow, travel_m=travel, tilt_deg=tilt,
                confidence=0.6, source="rules",
            ).validated()
    return ActionPlan(source="default", confidence=0.3).validated()
