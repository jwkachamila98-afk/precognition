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
        model_complexity: int = 0,  # 0 = fastest / lightest CPU profile for Intel Mac
        horizontal_fov_deg: float = 60.0,
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Run `pip install mediapipe` to use MediaPipeHandTracker."
            )

        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        # Needed to turn an apparent hand size in pixels into a distance. Most
        # laptop webcams sit near 60 degrees horizontally; an error here scales
        # the recovered depth proportionally but does not distort the hand.
        self.horizontal_fov_deg = float(horizontal_fov_deg)

        self._mp_hands = mp.solutions.hands
        self._tracker = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )


    # A hand nearer than this is touching the lens; further and it is out of the
    # workspace. Outside the band the size-based estimate is not believable.
    _MIN_DEPTH_M = 0.15
    _MAX_DEPTH_M = 1.60
    _NOMINAL_DEPTH_M = 0.50

    def _anchor_in_camera_frame(
        self, local: np.ndarray, kpts_2d: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Place hand-centred metric landmarks at their true camera-frame position.

        MediaPipe hands its metric landmarks back relative to the hand's own
        geometric centre; where the hand IS in the scene has to be recovered
        separately, and it is exactly what the reward and the co-adaptation
        signal need.

        This is a pose-from-correspondences problem: the metric shape is known,
        its projection is observed, and the rigid transform between them is
        wanted - so it is solved as one. A simpler apparent-size estimate
        (Z = f x metres / pixels) was tried first and is biased, because it
        assumes the measured span lies on the optical axis; off to one side it
        over-estimated depth by up to 20 cm at 40 cm off-centre, which is the
        very regime that matters here.
        """
        f_px = (0.5 * width) / np.tan(np.radians(self.horizontal_fov_deg) * 0.5)
        K = np.array([[f_px, 0.0, width * 0.5],
                      [0.0, f_px, height * 0.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        ok, rvec, tvec = False, None, None
        try:
            ok, rvec, tvec = cv2.solvePnP(
                local.astype(np.float64), kpts_2d.astype(np.float64), K, None,
                flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            ok = False

        depth_ok = tvec is not None and self._MIN_DEPTH_M <= float(tvec.reshape(-1)[2]) <= self._MAX_DEPTH_M
        if ok and depth_ok:
            R, _ = cv2.Rodrigues(rvec)
            return (local.astype(np.float64) @ R.T + tvec.reshape(1, 3)).astype(np.float32)

        # No usable solution - the hand is too foreshortened, too small or too
        # blurred to localise. Sit it at the nominal working distance on the ray
        # through its own centre, which is still better than the optical axis.
        centre_px = kpts_2d.mean(axis=0)
        depth = self._NOMINAL_DEPTH_M
        out = local.copy()
        out[:, 0] += (float(centre_px[0]) - width * 0.5) * depth / f_px
        out[:, 1] += (float(centre_px[1]) - height * 0.5) * depth / f_px
        out[:, 2] += depth
        return out.astype(np.float32)

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
                local = np.empty((21, 3), dtype=np.float32)
                for j_idx, wlm in enumerate(world_lms.landmark):
                    # MediaPipe world coordinates: X right, Y up, Z forward, and
                    # crucially they are relative to the HAND's own centre, not
                    # the camera. Convert to the camera convention (Y down).
                    local[j_idx] = (wlm.x, -wlm.y, wlm.z)
                local -= local.mean(axis=0)

                # Anchor the hand where it actually is. Copying the centred
                # world landmarks through and pasting a constant 0.5 m into z -
                # as this did - pins every hand to the optical axis and throws
                # its position away, keeping only shape and orientation. Any
                # comparison against a plan authored at the object then measures
                # where the OBJECT is rather than what the user did: a perfectly
                # executed reach still scored as a total failure whenever the
                # object sat off-centre, which is why episode reward was stuck
                # at -1.000 and the policy had no gradient to learn from.
                kpts_3d = self._anchor_in_camera_frame(local, kpts_2d, w, h)
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
