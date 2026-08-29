"""Ask Gemini to author a lathe profile for the object actually in view.

Gemini cannot emit geometry - no model here returns a mesh - so what it returns
is PARAMETERS and the mesh is built from them locally: a radius/height profile
in centimetres, which src.simulation.render.primitives.lathe turns into a
surface of revolution.

Why this beats the canonical per-class profiles it falls back to: those describe
a class, not an object. Every bottle gets the same shoulder and the same neck,
so a wine bottle, a squat sauce bottle and a sports flask are staged
identically. A profile authored from the crop describes the thing on the bench.

Why it is bounded so tightly: a malformed profile is worse than a generic one -
it renders as a spike or an inside-out solid, and unlike a wrong size it is not
obviously wrong, it just looks broken. Every response is validated for
monotonic heights, positive radii, plausible proportions and sane point counts,
and anything that fails is discarded in favour of the canonical shape.
"""

from __future__ import annotations

import base64
import json
import logging
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import certifi
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_PROMPT = (
    "This image shows a single object a person is about to pick up.\n"
    "Model it as a SURFACE OF REVOLUTION about its own upright axis.\n"
    "Reply with ONLY a JSON object, no prose and no code fence:\n"
    '{"shape":"<one word>","height_cm":<number>,'
    '"profile":[[radius_cm,height_cm],...],"upright":<true|false>}\n'
    "Rules: profile runs bottom to top; heights strictly increasing, starting at 0; "
    "radii > 0 except optionally at the very first or last point to close a dome; "
    "between 4 and 14 points; use real-world centimetres for a typical example of "
    "this object. Set upright false if the object is normally seen lying down "
    "(a remote, a pen, a book). Capture what makes THIS object's silhouette "
    "distinctive - a neck, a taper, a foot, a bulge."
)

_MIN_POINTS, _MAX_POINTS = 4, 14
_MIN_HEIGHT_CM, _MAX_HEIGHT_CM = 1.5, 60.0
_MAX_ASPECT = 12.0          # height : diameter, either way round


class GeminiMeshAuthor:
    """Requests a lathe profile for a specific object. Best-effort by design."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash",
                 timeout: float = 15.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def author(self, sprite: np.ndarray, label: str) -> Optional[List[List[float]]]:
        """Return a validated [[radius_cm, height_cm], ...] profile, or None."""
        if sprite is None or sprite.size == 0 or min(sprite.shape[:2]) < 8:
            return None
        try:
            payload = self._request(sprite, label)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            logger.warning(f"GeminiMeshAuthor: request failed ({exc}).")
            return None
        return self._validate(payload, label)

    def _request(self, sprite: np.ndarray, label: str) -> Dict[str, Any]:
        ok, buf = cv2.imencode(".jpg", sprite, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise ValueError("could not encode the crop")

        body = json.dumps({
            "contents": [{"parts": [
                {"text": f"{_PROMPT}\n\nThe object is described as: {label}."},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(buf.tobytes()).decode()}},
            ]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
        }).encode()

        req = urllib.request.Request(
            _ENDPOINT.format(model=self.model),
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())

        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)

    @staticmethod
    def _validate(payload: Dict[str, Any], label: str) -> Optional[List[List[float]]]:
        """Reject anything that would render as a spike or an inside-out solid.

        Deliberately strict. The fallback is a generic but correct shape, which
        is a far better failure than believable-looking garbage.
        """
        raw = payload.get("profile")
        if not isinstance(raw, list) or not (_MIN_POINTS <= len(raw) <= _MAX_POINTS):
            logger.info(f"GeminiMeshAuthor: '{label}' profile rejected (point count).")
            return None

        try:
            prof = [[float(r), float(h)] for r, h in raw]
        except (TypeError, ValueError):
            logger.info(f"GeminiMeshAuthor: '{label}' profile rejected (non-numeric).")
            return None

        heights = [h for _, h in prof]
        radii = [r for r, _ in prof]

        if any(b <= a for a, b in zip(heights, heights[1:])):
            logger.info(f"GeminiMeshAuthor: '{label}' profile rejected (heights not increasing).")
            return None
        if abs(heights[0]) > 1e-6:
            prof = [[r, h - heights[0]] for r, h in prof]      # re-base to zero
            heights = [h - heights[0] for h in heights]
        if any(r < 0.0 for r in radii) or max(radii) <= 0.0:
            logger.info(f"GeminiMeshAuthor: '{label}' profile rejected (radii).")
            return None
        if any(r <= 0.0 for r in radii[1:-1]):
            logger.info(f"GeminiMeshAuthor: '{label}' profile rejected (pinched to zero mid-body).")
            return None

        height = heights[-1]
        diameter = 2.0 * max(radii)
        if not (_MIN_HEIGHT_CM <= height <= _MAX_HEIGHT_CM):
            logger.info(f"GeminiMeshAuthor: '{label}' profile rejected (height {height:.1f} cm).")
            return None
        if max(height / diameter, diameter / height) > _MAX_ASPECT:
            logger.info(f"GeminiMeshAuthor: '{label}' profile rejected (aspect).")
            return None

        logger.info(
            f"GeminiMeshAuthor: authored a {len(prof)}-point profile for '{label}' "
            f"({diameter:.1f} cm across x {height:.1f} cm tall)."
        )
        return prof
