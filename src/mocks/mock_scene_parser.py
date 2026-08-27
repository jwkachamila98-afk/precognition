"""Synthetic 3D Scene Parser with Dynamic Intent-Grounded Object Extraction."""

import math
import re
import time
from typing import List, Optional
import numpy as np

from src.perception.depth_estimator import DepthMap
from src.perception.scene_parser import (
    BoundingBox3D,
    ParsedScene,
    ScenePointCloud,
    SceneParserABC,
)


class MockSceneParser(SceneParserABC):
    """
    Simulates a language-conditioned 3D spatial scene parser (e.g. Lang-SAM / Grounding DINO + 3D Bounding).
    Dynamically extracts objects from natural language intent strings.
    If no object is specified or intent is idle/empty, returns 0 bounding boxes.
    """

    def __init__(self, num_points: int = 400) -> None:
        self.num_points = num_points
        self._start_time = time.time()

        # Database of canonical physical dimensions [dx, dy, dz] in meters
        self._object_database = {
            "remote": {
                "label": "remote_control",
                "size": np.array([0.06, 0.18, 0.03], dtype=np.float32), # 6x18x3 cm
                "center_offset": np.array([0.06, 0.10, 0.58], dtype=np.float32)
            },
            "cup": {
                "label": "coffee_mug",
                "size": np.array([0.09, 0.09, 0.11], dtype=np.float32), # 9x9x11 cm
                "center_offset": np.array([-0.08, 0.08, 0.55], dtype=np.float32)
            },
            "mug": {
                "label": "coffee_mug",
                "size": np.array([0.09, 0.09, 0.11], dtype=np.float32),
                "center_offset": np.array([-0.08, 0.08, 0.55], dtype=np.float32)
            },
            "bottle": {
                "label": "water_bottle",
                "size": np.array([0.08, 0.08, 0.22], dtype=np.float32), # 8x8x22 cm
                "center_offset": np.array([0.12, 0.06, 0.62], dtype=np.float32)
            },
            "pen": {
                "label": "stylus_pen",
                "size": np.array([0.015, 0.15, 0.015], dtype=np.float32),
                "center_offset": np.array([0.02, 0.12, 0.52], dtype=np.float32)
            },
            "stylus": {
                "label": "stylus_pen",
                "size": np.array([0.015, 0.15, 0.015], dtype=np.float32),
                "center_offset": np.array([0.02, 0.12, 0.52], dtype=np.float32)
            },
            "phone": {
                "label": "smartphone",
                "size": np.array([0.075, 0.15, 0.01], dtype=np.float32),
                "center_offset": np.array([0.04, 0.11, 0.54], dtype=np.float32)
            },
            "apple": {
                "label": "apple",
                "size": np.array([0.08, 0.08, 0.08], dtype=np.float32),
                "center_offset": np.array([-0.04, 0.09, 0.56], dtype=np.float32)
            },
            "box": {
                "label": "cardboard_box",
                "size": np.array([0.14, 0.12, 0.10], dtype=np.float32),
                "center_offset": np.array([0.00, 0.07, 0.60], dtype=np.float32)
            }
        }

    def _extract_object_from_intent(self, intent: str) -> Optional[dict]:
        """
        Extract target object geometry from intent string.
        Returns None if intent is empty, idle, or contains no target object.
        """
        if not intent or intent.strip().lower() in ("none", "idle", "clear", "off", ""):
            return None

        intent_clean = intent.lower()

        # 1. Direct keyword lookup
        for key, template in self._object_database.items():
            if key in intent_clean:
                return template

        # 2. Heuristic noun extraction (e.g. "pick up the [noun]")
        match = re.search(r"(?:pick\s+up|grasp|grab|take|get|reach\s+for|lift|hold)\s+(?:the\s+|a\s+|an\s+|this\s+)?([a-zA-Z_-]+)", intent_clean)
        if match:
            noun = match.group(1).lower()
            if noun not in ("it", "object", "item", "something", "up"):
                return {
                    "label": noun,
                    "size": np.array([0.08, 0.10, 0.06], dtype=np.float32),
                    "center_offset": np.array([0.05, 0.10, 0.56], dtype=np.float32)
                }

        return None

    def parse_scene(
        self,
        image: np.ndarray,
        depth: DepthMap,
        intent: str = "foresee me picking this remote control",
        intrinsics: Optional[np.ndarray] = None
    ) -> ParsedScene:
        """
        Synthesize 3D bounding boxes conditioned on user intent.
        If no target object is mentioned in intent, returns empty bounding_boxes list.
        """
        now = time.time()
        t = now - self._start_time
        target_template = self._extract_object_from_intent(intent)

        bboxes: List[BoundingBox3D] = []

        if target_template is not None:
            # Subtle natural micro-movement
            center = target_template["center_offset"].copy()
            center[0] += 0.01 * math.sin(t * 0.5)
            center[1] += 0.005 * math.cos(t * 0.7)

            # Slight rotation
            yaw = 0.15 * math.sin(t * 0.8)
            rotation = np.array([0.0, 0.0, yaw], dtype=np.float32)

            bbox = BoundingBox3D(
                label=target_template["label"],
                center=center,
                size=target_template["size"],
                rotation=rotation,
                confidence=0.94
            )
            bboxes.append(bbox)

        # Generate lightweight 3D point cloud for table plane & object (if present)
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

        point_cloud = ScenePointCloud(
            points=all_pts,
            colors=all_cols,
            timestamp=now
        )

        return ParsedScene(
            intent=intent,
            bounding_boxes=bboxes,
            point_cloud=point_cloud,
            timestamp=now
        )
