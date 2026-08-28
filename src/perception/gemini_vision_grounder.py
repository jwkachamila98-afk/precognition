"""Open-vocabulary visual grounding via Gemini 2.5 Flash.

Fallback for object references that don't resolve to a known COCO-80 class or need
spatial/visual disambiguation (e.g. "the red stapler", "the cup on the left") - cases
the local YOLO detector's fixed 80-class vocabulary structurally can't handle. Gated
by the caller to only fire on intent change, never per-frame, since this is a real
network API call with real latency and cost, unlike the local GPU detectors.
"""

import base64
import json
import logging
import ssl
import urllib.request
from typing import Optional

import certifi
import cv2
import numpy as np

from src.perception.object_detector import Detection2D

logger = logging.getLogger(__name__)

_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_PROMPT_TEMPLATE = (
    'Locate the object described as "{description}" in this image. '
    "Respond with ONLY a JSON object, no markdown fencing, in exactly this form: "
    '{{"found": true, "label": "short object name", "box_2d": [ymin, xmin, ymax, xmax]}}. '
    "box_2d values are integers from 0 to 1000, normalized to the image height/width "
    '(Gemini\'s standard detection convention). If the object is not visible in the '
    'image, respond with {{"found": false}} and omit box_2d.'
)


class GeminiVisionGrounder:
    """Open-vocabulary object grounding using Gemini's native bounding-box output."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash", timeout: float = 6.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def ground(self, image: np.ndarray, description: str) -> Optional[Detection2D]:
        """Ask Gemini to locate `description` in `image` (BGR, as produced by OpenCV).
        Returns a Detection2D in pixel coordinates, or None if not found / on any
        failure - never raises, since this is a best-effort fallback."""
        h, w = image.shape[:2]
        try:
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            payload = {
                "contents": [{
                    "parts": [
                        {"text": _PROMPT_TEMPLATE.format(description=description)},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                    ]
                }],
                "generationConfig": {"responseMimeType": "application/json"},
            }

            url = f"{_ENDPOINT_TEMPLATE.format(model=self.model)}?key={self.api_key}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = result["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(text)

            if not data.get("found") or "box_2d" not in data:
                logger.info(f"GeminiVisionGrounder: '{description}' not found in frame.")
                return None

            ymin, xmin, ymax, xmax = data["box_2d"]
            detection = Detection2D(
                label=str(data.get("label", description)),
                score=0.75,
                xmin=int(np.clip(xmin / 1000.0 * w, 0, w - 1)),
                ymin=int(np.clip(ymin / 1000.0 * h, 0, h - 1)),
                xmax=int(np.clip(xmax / 1000.0 * w, 0, w)),
                ymax=int(np.clip(ymax / 1000.0 * h, 0, h)),
            )
            logger.info(f"GeminiVisionGrounder: grounded '{description}' -> {detection}")
            return detection
        except Exception as e:
            logger.warning(f"GeminiVisionGrounder: grounding failed ({e}).")
            return None
