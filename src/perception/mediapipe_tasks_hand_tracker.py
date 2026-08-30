"""Live 3D hand tracker using the MediaPipe Tasks API (HandLandmarker).

Functionally equivalent to MediaPipeHandTracker, but built on the Tasks API instead of
the legacy `mp.solutions` API, which was removed in mediapipe 1.x. This is the tracker
that actually runs on the cloud GPU pod, where pip installs the latest mediapipe release.
"""

import logging
import ssl
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import certifi
import cv2
import numpy as np

from src.perception.hand_anchoring import DEFAULT_HFOV_DEG, anchor_hand

from src.perception.hand_tracker import (
    HandPose,
    HandSide,
    HandTrackerABC,
    MANOParameters,
)

logger = logging.getLogger(__name__)

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "models" / "hand_landmarker.task"


class MediaPipeTasksHandTracker(HandTrackerABC):
    """Live real-time hand tracker running on the MediaPipe Tasks API (HandLandmarker)."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
        horizontal_fov_deg: float = DEFAULT_HFOV_DEG,
    ) -> None:
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        model_path = self._ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    @staticmethod
    def _ensure_model() -> Path:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _MODEL_PATH.exists() or _MODEL_PATH.stat().st_size < 1024:
            logger.info(f"Downloading HandLandmarker model to {_MODEL_PATH}...")
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(_MODEL_URL, context=ctx, timeout=30) as resp, open(_MODEL_PATH, "wb") as f:
                f.write(resp.read())
        return _MODEL_PATH

    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None
    ) -> List[HandPose]:
        h, w = image.shape[:2]
        now = time.time()

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 and image.shape[2] == 3 else image
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        poses: List[HandPose] = []

        for idx, hand_lms in enumerate(result.hand_landmarks):
            kpts_2d = np.zeros((21, 2), dtype=np.float32)
            for j_idx, lm in enumerate(hand_lms):
                kpts_2d[j_idx, 0] = lm.x * w
                kpts_2d[j_idx, 1] = lm.y * h

            kpts_3d = np.zeros((21, 3), dtype=np.float32)
            if result.hand_world_landmarks and idx < len(result.hand_world_landmarks):
                world_lms = result.hand_world_landmarks[idx]
                local = np.empty((21, 3), dtype=np.float32)
                for j_idx, wlm in enumerate(world_lms):
                    # World landmarks are metric but relative to the HAND's own
                    # centre. Convert to the camera convention (Y down).
                    local[j_idx] = (wlm.x, -wlm.y, wlm.z)
                local -= local.mean(axis=0)
                # Recover where the hand actually is. Pasting a constant depth
                # here - as this did - pins every hand to the optical axis and
                # discards its position, which silently wrecks every comparison
                # against a plan authored at the object. See hand_anchoring.
                kpts_3d = anchor_hand(local, kpts_2d, w, h, self.horizontal_fov_deg)
            else:
                for j_idx, lm in enumerate(hand_lms):
                    kpts_3d[j_idx, 0] = (lm.x - 0.5) * 0.4
                    kpts_3d[j_idx, 1] = (lm.y - 0.5) * 0.4
                    kpts_3d[j_idx, 2] = 0.50 + lm.z * 0.2

            side = HandSide.RIGHT
            confidence = 0.90
            if result.handedness and idx < len(result.handedness):
                category = result.handedness[idx][0]
                raw_label = category.category_name.lower()
                side = HandSide.LEFT if raw_label == "left" else HandSide.RIGHT
                confidence = float(category.score)

            wrist_pos = kpts_3d[0].copy()
            palm_vec = kpts_3d[9] - kpts_3d[0]
            norm_palm = np.linalg.norm(palm_vec) + 1e-6
            palm_dir = palm_vec / norm_palm

            pitch = float(np.arctan2(palm_dir[1], palm_dir[2]))
            yaw = float(np.arctan2(palm_dir[0], palm_dir[2]))
            roll = 0.0

            mano_params = MANOParameters(
                wrist_rotation=np.array([pitch, yaw, roll], dtype=np.float32),
                wrist_translation=wrist_pos,
                joint_rotations=np.zeros(45, dtype=np.float32),
                shape_betas=np.zeros(10, dtype=np.float32)
            )

            poses.append(HandPose(
                hand_id=idx,
                side=side,
                keypoints_3d=kpts_3d,
                keypoints_2d=kpts_2d,
                confidence=confidence,
                timestamp=now,
                mano_params=mano_params
            ))

        return poses

    def reset(self) -> None:
        pass
