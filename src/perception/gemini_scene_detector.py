"""Open-vocabulary scene detection via Gemini (src/perception/gemini_scene_detector.py).

YOLO can only name the eighty COCO classes, so anything outside that list either
went undetected or was forced onto the nearest wrong label. Gemini names
whatever is actually there - but a call takes on the order of a second and is
billed, so it cannot run per frame.

This runs it on a CADENCE, on a background thread, and TRACKS the returned boxes
locally in between. The detector therefore refreshes its understanding of the
scene every second or two while the boxes it hands back stay glued to the
objects at full frame rate. The render loop never blocks on the network: it
always reads the most recent completed result.
"""

from __future__ import annotations

import base64
import json
import logging
import ssl
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import certifi
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_PROMPT = (
    "List the distinct physical objects a person could pick up or manipulate in "
    "this image. Ignore people, body parts, walls, floors and background. "
    "Respond with ONLY a JSON array, no markdown fencing, each element exactly: "
    '{{"label": "short common name", "box_2d": [ymin, xmin, ymax, xmax]}}. '
    "box_2d values are integers 0-1000 normalised to image height/width. "
    "Order them most prominent first and return at most {max_objects}."
    "{hint}"
)


@dataclass
class DetectedObject:
    """One object located in the frame, in pixel coordinates."""
    label: str
    box: Tuple[int, int, int, int]          # x1, y1, x2, y2
    confidence: float = 0.85
    tracked: bool = False                   # True if carried by the tracker, not fresh
    age_sec: float = 0.0

    @property
    def centre(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


class GeminiSceneDetector:
    """Cadence-driven open-vocabulary detection with local tracking between calls."""

    # Tracker choice is a speed decision, measured at 640x480 on one object:
    # MOSSE 0.46 ms, MedianFlow 0.82 ms, KCF 6.2 ms, CSRT 30 ms, MIL 37 ms.
    # This runs on every object every frame in a loop that is already tight, so
    # KCF's 6 ms x 6 objects (37 ms) would have cost more than the entire HUD.
    # MOSSE handles scale poorly, which does not matter here: it only has to
    # hold a box for the few seconds between detections, and every detection
    # re-seeds it from scratch.
    _TRACKERS = ("MOSSE", "MedianFlow", "KCF")

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash",
                 cadence_sec: float = 1.5, timeout: float = 15.0,
                 max_objects: int = 6) -> None:
        self.api_key = api_key
        self.model = model
        self.cadence_sec = float(cadence_sec)
        self.timeout = float(timeout)
        self.max_objects = int(max_objects)
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        self._lock = threading.Lock()
        self._objects: List[DetectedObject] = []
        self._trackers: Dict[str, object] = {}
        self._inflight = False
        self._last_request_t = 0.0
        self._last_success_t = 0.0
        self._consecutive_failures = 0

    # ---------------------------------------------------------------- public

    def update(self, frame: np.ndarray, hint: Optional[str] = None) -> List[DetectedObject]:
        """Advance the trackers and, if due, start a fresh detection.

        Never blocks: returns whatever is currently known about the scene.
        """
        self._advance_trackers(frame)
        if self._should_request():
            self._start_request(frame, hint)
        with self._lock:
            return list(self._objects)

    def find(self, name: str) -> Optional[DetectedObject]:
        """The detected object whose label best matches `name`."""
        if not name:
            return None
        want = name.lower().strip()
        with self._lock:
            objects = list(self._objects)
        if not objects:
            return None
        exact = [o for o in objects if o.label.lower() == want]
        if exact:
            return exact[0]
        # Then containment either way ("cup" matches "coffee cup", and vice versa).
        partial = [o for o in objects
                   if want in o.label.lower() or o.label.lower() in want]
        if partial:
            return partial[0]
        # Finally, any shared significant word.
        words = {w for w in want.split() if len(w) > 2}
        for obj in objects:
            if words & {w for w in obj.label.lower().split() if len(w) > 2}:
                return obj
        return None

    @property
    def healthy(self) -> bool:
        """False once detection has been failing long enough to distrust."""
        return self._consecutive_failures < 3

    @property
    def seconds_since_detection(self) -> float:
        return time.time() - self._last_success_t if self._last_success_t else float("inf")

    # --------------------------------------------------------------- internal

    def _should_request(self) -> bool:
        if self._inflight:
            return False
        return (time.time() - self._last_request_t) >= self.cadence_sec

    def _start_request(self, frame: np.ndarray, hint: Optional[str]) -> None:
        self._inflight = True
        self._last_request_t = time.time()
        snapshot = frame.copy()
        threading.Thread(target=self._request_worker, args=(snapshot, hint),
                         daemon=True, name="gemini-detect").start()

    def _request_worker(self, frame: np.ndarray, hint: Optional[str]) -> None:
        try:
            found = self._detect(frame, hint)
            if found is not None:
                self._adopt(frame, found)
                self._consecutive_failures = 0
                self._last_success_t = time.time()
            else:
                self._consecutive_failures += 1
        except Exception as exc:
            self._consecutive_failures += 1
            logger.debug(f"GeminiSceneDetector: detection failed ({exc}).")
        finally:
            self._inflight = False

    def _detect(self, frame: np.ndarray, hint: Optional[str]) -> Optional[List[DetectedObject]]:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        hint_text = ""
        if hint:
            hint_text = (f' The user is interested in "{hint}" - make sure it is '
                         "included if visible.")
        payload = {
            "contents": [{"parts": [
                {"text": _PROMPT.format(max_objects=self.max_objects, hint=hint_text)},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(buf.tobytes()).decode("ascii")}},
            ]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        url = f"{_ENDPOINT.format(model=self.model)}?key={self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse(text, frame.shape[:2])

    def _parse(self, text: str, shape: Tuple[int, int]) -> List[DetectedObject]:
        """Turn Gemini's 0-1000 normalised boxes into pixel boxes."""
        height, width = shape
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            data = data.get("objects", data.get("detections", []))
        out: List[DetectedObject] = []
        for item in data if isinstance(data, list) else []:
            try:
                label = str(item["label"]).strip()
                ymin, xmin, ymax, xmax = (float(v) for v in item["box_2d"])
            except (KeyError, TypeError, ValueError):
                continue
            x1 = int(round(xmin / 1000.0 * width))
            x2 = int(round(xmax / 1000.0 * width))
            y1 = int(round(ymin / 1000.0 * height))
            y2 = int(round(ymax / 1000.0 * height))
            x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
            if label and (x2 - x1) >= 4 and (y2 - y1) >= 4:
                out.append(DetectedObject(label=label, box=(x1, y1, x2, y2)))
        return out[: self.max_objects]

    def _make_tracker(self):
        """The fastest tracker this OpenCV build actually provides."""
        legacy = getattr(cv2, "legacy", None)
        for name in self._TRACKERS:
            for factory in (getattr(legacy, f"Tracker{name}_create", None),
                            getattr(cv2, f"Tracker{name}_create", None)):
                if factory is None:
                    continue
                try:
                    return factory()
                except Exception:
                    continue
        return None

    def _adopt(self, frame: np.ndarray, found: List[DetectedObject]) -> None:
        """Replace the known scene and re-seed a tracker per object."""
        trackers: Dict[str, object] = {}
        for obj in found:
            tracker = self._make_tracker()
            if tracker is None:
                continue
            x1, y1, x2, y2 = obj.box
            try:
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                trackers[obj.label] = tracker
            except Exception:
                pass
        with self._lock:
            self._objects = found
            self._trackers = trackers

    def _advance_trackers(self, frame: np.ndarray) -> None:
        """Carry each box forward one frame so it stays on its object."""
        with self._lock:
            objects, trackers = list(self._objects), dict(self._trackers)
        if not objects:
            return
        age = self.seconds_since_detection
        updated: List[DetectedObject] = []
        for obj in objects:
            tracker = trackers.get(obj.label)
            box = obj.box
            tracked = obj.tracked
            if tracker is not None:
                try:
                    ok, bbox = tracker.update(frame)
                except Exception:
                    ok, bbox = False, None
                if ok and bbox is not None:
                    x, y, w, h = (int(round(v)) for v in bbox)
                    if w >= 4 and h >= 4:
                        box = (x, y, x + w, y + h)
                        tracked = True
            updated.append(DetectedObject(label=obj.label, box=box,
                                          confidence=obj.confidence,
                                          tracked=tracked, age_sec=age))
        with self._lock:
            self._objects = updated


class GeminiObjectDetector:
    """`ObjectDetectorABC` adapter, so the open-vocabulary detector can stand in
    wherever YOLO does.

    `detect` is called once per frame by the scene parser and must be cheap: it
    advances the trackers and returns the current understanding of the scene,
    starting a fresh Gemini call only when one is due.
    """

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash",
                 cadence_sec: float = 1.5, timeout: float = 15.0,
                 max_objects: int = 6, score_threshold: float = 0.0) -> None:
        self.scene = GeminiSceneDetector(api_key=api_key, model=model,
                                        cadence_sec=cadence_sec, timeout=timeout,
                                        max_objects=max_objects)
        self.score_threshold = float(score_threshold)
        self._hint: Optional[str] = None

    def set_hint(self, description: Optional[str]) -> None:
        """Name the object the user cares about, so it is not crowded out of a
        response capped at a handful of objects."""
        self._hint = description or None

    def detect(self, image: np.ndarray):
        from src.perception.object_detector import Detection2D
        objects = self.scene.update(image, hint=self._hint)
        out = []
        for obj in objects:
            if obj.confidence < self.score_threshold:
                continue
            x1, y1, x2, y2 = obj.box
            out.append(Detection2D(label=obj.label, score=obj.confidence,
                                   xmin=int(x1), ymin=int(y1),
                                   xmax=int(x2), ymax=int(y2)))
        return out
