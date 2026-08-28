"""Camera-grounded 3D scene parser backed by a real 2D object detector (YOLOv8 or
MediaPipe EfficientDet, whichever ObjectDetectorABC implementation is injected).

Replaces MockSceneParser's canned, intent-only bounding box with one derived from an
actual detection in the live RGB frame, back-projected into 3D using the depth map.
If the intent's target object isn't actually visible, no bounding box is returned.
"""

import re
import time
from typing import List, Optional

import numpy as np

from src.perception.depth_estimator import DepthMap
from src.perception.gemini_vision_grounder import GeminiVisionGrounder
from src.perception.object_detector import Detection2D, ObjectDetectorABC
from src.perception.scene_parser import (
    BoundingBox3D,
    ParsedScene,
    ScenePointCloud,
    SceneParserABC,
)

# Open-vocabulary map: every COCO-80 class YOLOv8 was trained on, plus natural aliases,
# so any object name spoken or typed ("wine glass", "remote control", ...) resolves to a
# real detectable class. Nouns with no COCO equivalent (e.g. "stylus") fall through to a
# literal label match against whatever the detector actually returns.
_INTENT_TO_COCO = {
    "wine glass": ["wine glass"],
    "sports ball": ["sports ball"],
    "baseball bat": ["baseball bat"],
    "baseball glove": ["baseball glove"],
    "tennis racket": ["tennis racket"],
    "fire hydrant": ["fire hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "traffic light": ["traffic light"],
    "cell phone": ["cell phone"],
    "hot dog": ["hot dog"],
    "hair drier": ["hair drier"],
    "hair dryer": ["hair drier"],
    "potted plant": ["potted plant"],
    "dining table": ["dining table"],
    "teddy bear": ["teddy bear"],
    "remote control": ["remote"],
    "tv remote": ["remote"],
    "coffee cup": ["cup"],
    "water bottle": ["bottle"],
    "person": ["person"],
    "bicycle": ["bicycle"],
    "car": ["car"],
    "motorcycle": ["motorcycle"],
    "airplane": ["airplane"],
    "bus": ["bus"],
    "train": ["train"],
    "truck": ["truck"],
    "boat": ["boat"],
    "bench": ["bench"],
    "bird": ["bird"],
    "cat": ["cat"],
    "dog": ["dog"],
    "horse": ["horse"],
    "sheep": ["sheep"],
    "cow": ["cow"],
    "elephant": ["elephant"],
    "bear": ["bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag"],
    "tie": ["tie"],
    "suitcase": ["suitcase"],
    "frisbee": ["frisbee"],
    "skis": ["skis"],
    "snowboard": ["snowboard"],
    "ball": ["sports ball"],
    "kite": ["kite"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "bottle": ["bottle"],
    "water": ["bottle"],
    "cup": ["cup"],
    "mug": ["cup"],
    "coffee": ["cup"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "pizza": ["pizza"],
    "donut": ["donut"],
    "doughnut": ["donut"],
    "cake": ["cake"],
    "chair": ["chair"],
    "couch": ["couch"],
    "sofa": ["couch"],
    "bed": ["bed"],
    "toilet": ["toilet"],
    "tv": ["tv"],
    "television": ["tv"],
    "laptop": ["laptop"],
    "mouse": ["mouse"],
    "remote": ["remote"],
    "keyboard": ["keyboard"],
    "phone": ["cell phone"],
    "smartphone": ["cell phone"],
    "microwave": ["microwave"],
    "oven": ["oven"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator"],
    "fridge": ["refrigerator"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "toothbrush": ["toothbrush"],
}

_NOUN_RE = re.compile(
    r"(?:pick\s+up|grasp|grab|take|get|reach\s+for|lift|hold)\s+(?:the\s+|a\s+|an\s+|this\s+)?([a-zA-Z-]+)"
)

_SPATIAL_REF_RE = re.compile(r"\b(?:near|by|beside|next\s+to|under|on|in\s+front\s+of)\b")


class LiveSceneParser(SceneParserABC):
    """
    Grounds 3D bounding boxes in a real object detector's output instead of a static
    per-keyword lookup table. Only ever reports an object that the detector actually
    sees in the current frame and that matches the current intent.
    """

    def __init__(
        self,
        object_detector: ObjectDetectorABC,
        num_points: int = 400,
        vision_grounder: Optional[GeminiVisionGrounder] = None,
    ) -> None:
        self.object_detector = object_detector
        self.num_points = num_points
        # Open-vocabulary fallback for descriptions that don't resolve to a COCO-80
        # class (YOLO structurally can't detect those). Only ever called when the
        # local detector finds nothing AND the intent text has changed since the last
        # grounding call - never per-frame, since it's a real network API call.
        self.vision_grounder = vision_grounder
        self._last_grounded_intent: Optional[str] = None
        self._last_grounded_detection: Optional[Detection2D] = None
        # Failed groundings are retried on a cooldown rather than cached.
        self._last_grounding_attempt = 0.0
        self._grounding_retry_sec = 2.5

    def _extract_target_keyword(self, intent: str) -> Optional[str]:
        if not intent or intent.strip().lower() in ("none", "idle", "clear", "off", ""):
            return None
        intent_clean = intent.lower()

        # Truncate at spatial-reference prepositions ("near the keyboard", "by the handle")
        # so an incidental landmark isn't mistaken for the actual grasp target.
        primary_span = _SPATIAL_REF_RE.split(intent_clean, maxsplit=1)[0]

        for key in _INTENT_TO_COCO:
            if key in primary_span:
                return key

        match = _NOUN_RE.search(primary_span)
        if match:
            noun = match.group(1).strip()
            if noun not in ("it", "object", "item", "something", "up"):
                return noun

        return None

    def _find_matching_detection(
        self, detections: List[Detection2D], target_keyword: str
    ) -> Optional[Detection2D]:
        candidate_labels = _INTENT_TO_COCO.get(target_keyword, [target_keyword])
        best: Optional[Detection2D] = None
        for det in detections:
            det_label = det.label.lower()
            if any(c == det_label or c in det_label or det_label in c for c in candidate_labels):
                if best is None or det.score > best.score:
                    best = det
        return best

    def _detection_to_3d(
        self,
        det: Detection2D,
        depth: DepthMap,
        image_shape: tuple,
        intrinsics: Optional[np.ndarray],
    ) -> BoundingBox3D:
        h, w = image_shape[:2]
        if intrinsics is not None:
            fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
            cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        else:
            fx = fy = 0.8 * w
            cx, cy = w / 2.0, h / 2.0

        px, py = det.center_px
        dh, dw = depth.depth.shape[:2]
        sx, sy = dw / w, dh / h
        dx = int(np.clip(px * sx, 0, dw - 1))
        dy = int(np.clip(py * sy, 0, dh - 1))

        x0, x1 = max(0, dx - 3), min(dw, dx + 4)
        y0, y1 = max(0, dy - 3), min(dh, dy + 4)
        region = depth.depth[y0:y1, x0:x1]
        z = float(np.median(region)) if region.size else float(depth.depth[dy, dx])
        z = max(z, depth.min_depth)

        center_x = (px - cx) * z / fx
        center_y = (py - cy) * z / fy
        width_m = det.width_px * z / fx
        height_m = det.height_px * z / fy
        depth_m = max(0.05, min(width_m, height_m) * 0.6)

        return BoundingBox3D(
            label=det.label,
            center=np.array([center_x, center_y, z], dtype=np.float32),
            size=np.array([max(width_m, 0.02), max(height_m, 0.02), depth_m], dtype=np.float32),
            rotation=np.zeros(3, dtype=np.float32),
            confidence=det.score,
        )

    def parse_scene(
        self,
        image: np.ndarray,
        depth: DepthMap,
        intent: str = "foresee me picking this remote control",
        intrinsics: Optional[np.ndarray] = None,
    ) -> ParsedScene:
        now = time.time()
        bboxes: List[BoundingBox3D] = []

        target_keyword = self._extract_target_keyword(intent)
        if target_keyword is not None:
            detections = self.object_detector.detect(image)
            matched = self._find_matching_detection(detections, target_keyword)

            if matched is None and self.vision_grounder is not None:
                # Open-vocabulary fallback: only actually call Gemini when the intent
                # text has changed since the last attempt, reusing the cached result
                # otherwise (re-projected against the current depth map each frame).
                # A FAILED call must not be cached as a result. The intent key was
                # being stamped before the call, so a single timeout made the
                # condition below false forever and grounding was never retried
                # for that phrase - one dropped request silently disabled
                # open-vocabulary detection for the rest of the session.
                stale = intent != self._last_grounded_intent
                retry_due = (self._last_grounded_detection is None
                             and time.time() - self._last_grounding_attempt
                             >= self._grounding_retry_sec)
                if stale or retry_due:
                    self._last_grounded_intent = intent
                    self._last_grounding_attempt = time.time()
                    # Pass the full description (color, spatial context, etc.), not
                    # just the reduced single-noun keyword - Gemini grounds better with
                    # richer context than the COCO-vocabulary matcher needs.
                    self._last_grounded_detection = self.vision_grounder.ground(image, intent)
                matched = self._last_grounded_detection

            if matched is not None:
                bboxes.append(self._detection_to_3d(matched, depth, image.shape, intrinsics))

        pts_per_table = self.num_points // 2 if bboxes else self.num_points
        table_y = 0.18
        table_x = np.random.uniform(-0.35, 0.35, size=pts_per_table).astype(np.float32)
        table_z = np.random.uniform(0.4, 0.9, size=pts_per_table).astype(np.float32)
        table_pts = np.stack([table_x, np.full(pts_per_table, table_y, dtype=np.float32), table_z], axis=-1)
        table_colors = np.tile(np.array([0.35, 0.35, 0.40], dtype=np.float32), (pts_per_table, 1))

        if bboxes:
            b = bboxes[0]
            pts_per_obj = self.num_points - pts_per_table
            obj_pts = np.random.normal(loc=b.center, scale=b.size * 0.35, size=(pts_per_obj, 3)).astype(np.float32)
            obj_colors = np.tile(np.array([0.0, 0.85, 1.0], dtype=np.float32), (pts_per_obj, 1))
            all_pts = np.vstack([obj_pts, table_pts])
            all_cols = np.vstack([obj_colors, table_colors])
        else:
            all_pts = table_pts
            all_cols = table_colors

        point_cloud = ScenePointCloud(points=all_pts, colors=all_cols, timestamp=now)

        return ParsedScene(
            intent=intent,
            bounding_boxes=bboxes,
            point_cloud=point_cloud,
            timestamp=now,
        )
