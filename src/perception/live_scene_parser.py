"""Camera-grounded 3D scene parser backed by a real 2D object detector.

Replaces MockSceneParser's canned, intent-only bounding box with one derived from an
actual detection in the live RGB frame, back-projected into 3D using the depth map.
If the intent's target object isn't actually visible, no bounding box is returned.
"""

import re
import time
from typing import List, Optional

import numpy as np

from src.perception.depth_estimator import DepthMap
from src.perception.object_detector import Detection2D, ObjectDetectorABC
from src.perception.scene_parser import (
    BoundingBox3D,
    ParsedScene,
    ScenePointCloud,
    SceneParserABC,
)

# Map common intent nouns to the COCO-80 class name(s) EfficientDet-Lite0 was trained on.
# Nouns with no COCO equivalent (e.g. "stylus") fall through to a literal label match instead.
_INTENT_TO_COCO = {
    "remote": ["remote"],
    "cup": ["cup"],
    "mug": ["cup"],
    "coffee": ["cup"],
    "bottle": ["bottle"],
    "water": ["bottle"],
    "phone": ["cell phone"],
    "smartphone": ["cell phone"],
    "apple": ["apple"],
    "banana": ["banana"],
    "orange": ["orange"],
    "book": ["book"],
    "scissors": ["scissors"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "laptop": ["laptop"],
    "clock": ["clock"],
    "vase": ["vase"],
    "ball": ["sports ball"],
    "spoon": ["spoon"],
    "fork": ["fork"],
    "knife": ["knife"],
    "bowl": ["bowl"],
}

_NOUN_RE = re.compile(
    r"(?:pick\s+up|grasp|grab|take|get|reach\s+for|lift|hold)\s+(?:the\s+|a\s+|an\s+|this\s+)?([a-zA-Z-]+)"
)


class LiveSceneParser(SceneParserABC):
    """
    Grounds 3D bounding boxes in a real object detector's output instead of a static
    per-keyword lookup table. Only ever reports an object that the detector actually
    sees in the current frame and that matches the current intent.
    """

    def __init__(self, object_detector: ObjectDetectorABC, num_points: int = 400) -> None:
        self.object_detector = object_detector
        self.num_points = num_points

    def _extract_target_keyword(self, intent: str) -> Optional[str]:
        if not intent or intent.strip().lower() in ("none", "idle", "clear", "off", ""):
            return None
        intent_clean = intent.lower()

        for key in _INTENT_TO_COCO:
            if key in intent_clean:
                return key

        match = _NOUN_RE.search(intent_clean)
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
