"""Live CPU-friendly 3D Hand Tracker using MediaPipe Hands."""

import time
from typing import List, Optional
import cv2
import numpy as np

from src.perception.hand_tracker import (
    HandPose,
    HandSide,
    HandTrackerABC,
    MANOParameters,
)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class MediaPipeHandTracker(HandTrackerABC):
    """
    Live real-time hand tracker running on CPU.
    Extracts 21 2D pixel coordinates and 3D metric world keypoints from RGB webcam frames.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
        model_complexity: int = 0  # 0 = fastest / lightest CPU profile for Intel Mac
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Run `pip install mediapipe` to use MediaPipeHandTracker."
            )

        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        self._mp_hands = mp.solutions.hands
        self._tracker = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None
    ) -> List[HandPose]:
        """
        Estimate 2D/3D hand keypoints from a live RGB image frame.
        """
        h, w = image.shape[:2]
        now = time.time()

        # MediaPipe expects RGB format
        if image.ndim == 3 and image.shape[2] == 3:
            # Check if image is BGR from OpenCV (default) and convert to RGB
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = image

        results = self._tracker.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return []

        poses: List[HandPose] = []

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            # Extract 2D image coordinates (u, v)
            kpts_2d = np.zeros((21, 2), dtype=np.float32)
            for j_idx, lm in enumerate(hand_lms.landmark):
                kpts_2d[j_idx, 0] = lm.x * w
                kpts_2d[j_idx, 1] = lm.y * h

            # Extract 3D metric coordinates
            # Use multi_hand_world_landmarks if available (in meters)
            kpts_3d = np.zeros((21, 3), dtype=np.float32)
            if results.multi_hand_world_landmarks and idx < len(results.multi_hand_world_landmarks):
                world_lms = results.multi_hand_world_landmarks[idx]
                for j_idx, wlm in enumerate(world_lms.landmark):
                    # MediaPipe world coordinates: X right, Y up, Z forward (relative to hand center)
                    # Convert to camera convention: X right, Y down, Z forward
                    kpts_3d[j_idx, 0] = wlm.x
                    kpts_3d[j_idx, 1] = -wlm.y
                    # Place wrist approx 0.5m in front of camera
                    kpts_3d[j_idx, 2] = 0.50 + wlm.z
            else:
                # Fallback depth estimation from normalized landmarks
                for j_idx, lm in enumerate(hand_lms.landmark):
                    kpts_3d[j_idx, 0] = (lm.x - 0.5) * 0.4
                    kpts_3d[j_idx, 1] = (lm.y - 0.5) * 0.4
                    kpts_3d[j_idx, 2] = 0.50 + lm.z * 0.2

            # Extract handedness and confidence score
            side = HandSide.RIGHT
            confidence = 0.90

            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness_info = results.multi_handedness[idx].classification[0]
                # MediaPipe assumes unmirrored input, so "Left" label in webcam selfie is physical Right hand
                raw_label = handedness_info.label.lower()
                side = HandSide.LEFT if raw_label == "left" else HandSide.RIGHT
                confidence = float(handedness_info.score)

            # Approximate MANO parameters from keypoint geometry
            # Wrist vector
            wrist_pos = kpts_3d[0].copy()
            # Direction from wrist to middle MCP (joint 9)
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

            pose = HandPose(
                hand_id=idx,
                side=side,
                keypoints_3d=kpts_3d,
                keypoints_2d=kpts_2d,
                confidence=confidence,
                timestamp=now,
                mano_params=mano_params
            )
            poses.append(pose)

        return poses

    def reset(self) -> None:
        if self._tracker:
            self._tracker.reset()
