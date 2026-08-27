"""High-performance synthetic MANO hand tracker mock for CPU-only development."""

import math
import time
from typing import List, Optional
import numpy as np

from src.perception.hand_tracker import (
    HandPose,
    HandSide,
    HandTrackerABC,
    MANOParameters,
)


class MockHandTracker(HandTrackerABC):
    """
    Simulates a 3D hand estimator (e.g. Fast-HaMeR / MediaPipe) with zero GPU overhead.
    Computes articulated 21-joint forward kinematics and perspective projections.
    Runs at >300 FPS on Intel Mac CPUs.
    """

    def __init__(
        self,
        hand_side: HandSide = HandSide.RIGHT,
        base_depth: float = 0.55, # 55 cm in front of camera
        animate: bool = True
    ) -> None:
        self.hand_side = hand_side
        self.base_depth = base_depth
        self.animate = animate
        self._start_time = time.time()

        # Canonical relative joint offsets (in meters) from parent to child
        # Joint 0: Wrist at origin
        # Fingers: [Thumb, Index, Middle, Ring, Pinky]
        self._base_finger_roots = np.array([
            [-0.035, -0.02, 0.025], # 1: Thumb CMC
            [-0.030, -0.09, 0.010], # 5: Index MCP
            [-0.005, -0.095, 0.010],# 9: Middle MCP
            [ 0.020, -0.09, 0.010], # 13: Ring MCP
            [ 0.045, -0.08, 0.010], # 17: Pinky MCP
        ], dtype=np.float32)

        # Bone segment lengths for 3 phalanges per finger
        # [MCP->PIP, PIP->DIP, DIP->TIP]
        self._segment_lengths = {
            "thumb": [0.035, 0.030, 0.025],
            "index": [0.040, 0.028, 0.022],
            "middle": [0.045, 0.032, 0.024],
            "ring": [0.040, 0.028, 0.022],
            "pinky": [0.032, 0.022, 0.018],
        }

    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None
    ) -> List[HandPose]:
        """
        Synthesize 3D hand keypoints and 2D projections onto image frame.
        """
        h, w = image.shape[:2]
        now = time.time()
        t = (now - self._start_time) if self.animate else 0.0

        # Construct default pinhole intrinsics if not provided
        if intrinsics is None:
            fx = fy = 0.8 * w
            cx = w / 2.0
            cy = h / 2.0
            k_matrix = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            k_matrix = intrinsics
            fx = k_matrix[0, 0]
            fy = k_matrix[1, 1]
            cx = k_matrix[0, 2]
            cy = k_matrix[1, 2]

        # Dynamic wrist translation in camera frame (meters)
        # Gentle hovering motion
        wrist_x = 0.08 * math.sin(t * 1.2) + (0.05 if self.hand_side == HandSide.RIGHT else -0.05)
        wrist_y = 0.04 * math.cos(t * 0.9) + 0.05
        wrist_z = self.base_depth + 0.05 * math.sin(t * 0.6)
        wrist_pos = np.array([wrist_x, wrist_y, wrist_z], dtype=np.float32)

        # Dynamic wrist rotation (Euler angles)
        pitch = 0.15 * math.sin(t * 1.5)
        yaw = 0.20 * math.cos(t * 1.1)
        roll = 0.10 * math.sin(t * 0.8)

        # Construct 3D keypoints array: (21, 3)
        keypoints_3d = np.zeros((21, 3), dtype=np.float32)
        keypoints_3d[0] = wrist_pos # Joint 0: Wrist

        # Compute finger flexion angles (wave / grab oscillation)
        finger_flex = [
            0.3 + 0.2 * math.sin(t * 2.0),         # Thumb
            0.4 + 0.35 * math.sin(t * 2.0 + 0.2),  # Index
            0.4 + 0.35 * math.sin(t * 2.0 + 0.4),  # Middle
            0.4 + 0.35 * math.sin(t * 2.0 + 0.6),  # Ring
            0.4 + 0.35 * math.sin(t * 2.0 + 0.8),  # Pinky
        ]

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        joint_idx = 1

        for f_idx, f_name in enumerate(finger_names):
            root_offset = self._base_finger_roots[f_idx].copy()
            if self.hand_side == HandSide.LEFT:
                root_offset[0] *= -1.0 # Mirror X for left hand

            # Joint: MCP / CMC
            mcp_pos = wrist_pos + root_offset
            keypoints_3d[joint_idx] = mcp_pos
            prev_joint = mcp_pos
            joint_idx += 1

            flex = finger_flex[f_idx]
            lengths = self._segment_lengths[f_name]

            # 3 phalange segments per finger
            for seg_i, length in enumerate(lengths):
                # Direction vector extending outward and flexing inward along Y and Z
                cur_flex = flex * (seg_i + 1) * 0.7
                dx = 0.01 * math.sin(cur_flex)
                dy = -length * math.cos(cur_flex)
                dz = length * math.sin(cur_flex)

                new_joint = prev_joint + np.array([dx, dy, dz], dtype=np.float32)
                keypoints_3d[joint_idx] = new_joint
                prev_joint = new_joint
                joint_idx += 1

        # Project 3D keypoints to 2D image plane (u, v)
        # u = fx * (X / Z) + cx, v = fy * (Y / Z) + cy
        keypoints_2d = np.zeros((21, 2), dtype=np.float32)
        valid_z = np.clip(keypoints_3d[:, 2], 0.1, 10.0)
        keypoints_2d[:, 0] = fx * (keypoints_3d[:, 0] / valid_z) + cx
        keypoints_2d[:, 1] = fy * (keypoints_3d[:, 1] / valid_z) + cy

        # Synthesize MANO model parameters
        mano_params = MANOParameters(
            wrist_rotation=np.array([pitch, yaw, roll], dtype=np.float32),
            wrist_translation=wrist_pos,
            joint_rotations=np.full(45, 0.05 * math.sin(t), dtype=np.float32),
            shape_betas=np.zeros(10, dtype=np.float32)
        )

        hand_pose = HandPose(
            hand_id=0,
            side=self.hand_side,
            keypoints_3d=keypoints_3d,
            keypoints_2d=keypoints_2d,
            confidence=0.96 + 0.03 * math.sin(t),
            timestamp=now,
            mano_params=mano_params
        )

        return [hand_pose]

    def reset(self) -> None:
        self._start_time = time.time()
