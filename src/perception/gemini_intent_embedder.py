"""Turn what the user said into state the policy can actually learn from.

The 112-dimensional discrepancy state is entirely geometric - keypoint offsets,
wrist poses, contact, velocity, reward history. Nothing in it encodes the
INSTRUCTION. "Pick it up gently" and "grab it fast" therefore produce byte
identical states and, necessarily, identical corrections: the policy could not
condition on the words even in principle. The transcript was used to pick a
target noun and then discarded.

This embeds the whole utterance instead, so the manner of the request reaches
the learner along with the geometry.

Honest limits, because they matter for reading results:

  * The embedding is `gemini-embedding-001` truncated to 32 dimensions. Measured
    on short grasp instructions, the ORDERING is correct - "quickly"/"fast"
    score 0.977 against each other and "quickly"/"fragile" 0.898 - but every
    pair sits in a narrow 0.88-0.98 band. The discriminative part is a small
    fraction of the vector's magnitude, so intent-conditioned behaviour will
    take many episodes to emerge, not a handful.
  * It is opaque. When the policy changes behaviour after a differently-worded
    instruction, nothing here can say which words were responsible.
  * With no API key, or on any failure, the dimensions are zero and the policy
    behaves exactly as it did before - intent-blind, not broken.
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.request
from typing import Dict, Optional

import certifi
import numpy as np

logger = logging.getLogger(__name__)

_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/"
             "models/{model}:embedContent")

INTENT_DIMS = 32


class GeminiIntentEmbedder:
    """Embeds an utterance into INTENT_DIMS unit-norm dimensions. Cached per text."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001",
                 timeout: float = 12.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._cache: Dict[str, np.ndarray] = {}

    def cached(self, text: Optional[str]) -> Optional[np.ndarray]:
        """The embedding for `text` if it is already known, without a request."""
        if not text:
            return None
        return self._cache.get(text.strip().lower())

    def embed(self, text: Optional[str]) -> Optional[np.ndarray]:
        """Embed `text`, returning a cached vector when one exists.

        Blocking: callers that run inside a frame loop should do this off the
        hot path and use `cached()` there.
        """
        if not text or not text.strip():
            return None
        key = text.strip().lower()
        if key in self._cache:
            return self._cache[key]

        try:
            body = json.dumps({
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": INTENT_DIMS,
            }).encode()
            req = urllib.request.Request(
                _ENDPOINT.format(model=self.model), data=body,
                headers={"Content-Type": "application/json",
                         "x-goog-api-key": self.api_key},
            )
            with urllib.request.urlopen(req, context=self._ssl_ctx,
                                        timeout=self.timeout) as resp:
                values = json.loads(resp.read().decode())["embedding"]["values"]
        except (urllib.error.URLError, OSError, KeyError, ValueError, TimeoutError) as exc:
            logger.warning(f"GeminiIntentEmbedder: '{text[:40]}' failed ({exc}); "
                           f"the policy stays intent-blind for this utterance.")
            return None

        vec = np.asarray(values, dtype=np.float32)
        if vec.shape != (INTENT_DIMS,) or not np.all(np.isfinite(vec)):
            logger.warning(f"GeminiIntentEmbedder: malformed vector for '{text[:40]}'.")
            return None

        # Truncating 3072 dimensions to 32 leaves the vector well short of unit
        # norm (~0.18), so renormalise: the policy should see direction, not an
        # artefact of how many dimensions were kept.
        vec = vec / max(float(np.linalg.norm(vec)), 1e-9)
        self._cache[key] = vec
        logger.info(f"GeminiIntentEmbedder: embedded '{text[:48]}' "
                    f"({INTENT_DIMS} dims).")
        return vec
