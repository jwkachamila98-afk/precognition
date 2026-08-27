"""Perception interfaces and dataclasses for hand pose and 3D mesh estimation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


# Standard 21-joint MANO / MediaPipe kinematic topology
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross connections
    (5, 9), (9, 13), (13, 17)
]

JOINT_NAMES = [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]


@dataclass
class MANOParameters:
    """MANO parametric hand model coefficients."""
    wrist_rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    wrist_translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    joint_rotations: np.ndarray = field(default_factory=lambda: np.zeros(45, dtype=np.float32)) # 15 joints * 3 axis-angle
    shape_betas: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float32)) # PCA shape parameters

    def to_dict(self) -> dict:
        return {
            "wrist_rotation": self.wrist_rotation.tolist(),
            "wrist_translation": self.wrist_translation.tolist(),
            "joint_rotations": self.joint_rotations.tolist(),
            "shape_betas": self.shape_betas.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MANOParameters":
        return cls(
            wrist_rotation=np.array(data.get("wrist_rotation", np.zeros(3)), dtype=np.float32),
            wrist_translation=np.array(data.get("wrist_translation", np.zeros(3)), dtype=np.float32),
            joint_rotations=np.array(data.get("joint_rotations", np.zeros(45)), dtype=np.float32),
            shape_betas=np.array(data.get("shape_betas", np.zeros(10)), dtype=np.float32)
        )


# Alias for shorthand usage
ManoParams = MANOParameters


@dataclass
class HandPose:
    """Estimated 3D and 2D hand pose with joint keypoints and MANO parameters."""
    hand_id: int
    side: HandSide
    # 21 3D joint locations in camera coordinate frame (X-right, Y-down, Z-forward) in meters
    keypoints_3d: np.ndarray # shape: (21, 3)
    # 21 2D joint pixel coordinates in image frame (u, v)
    keypoints_2d: np.ndarray # shape: (21, 2)
    confidence: float
    timestamp: float
    mano_params: Optional[MANOParameters] = None

    def to_dict(self) -> dict:
        return {
            "hand_id": self.hand_id,
            "side": self.side.value,
            "keypoints_3d": self.keypoints_3d.tolist(),
            "keypoints_2d": self.keypoints_2d.tolist(),
            "confidence": float(self.confidence),
            "timestamp": float(self.timestamp),
            "mano_params": self.mano_params.to_dict() if self.mano_params else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HandPose":
        mano = MANOParameters.from_dict(data["mano_params"]) if data.get("mano_params") else None
        return cls(
            hand_id=int(data["hand_id"]),
            side=HandSide(data["side"]),
            keypoints_3d=np.array(data["keypoints_3d"], dtype=np.float32),
            keypoints_2d=np.array(data["keypoints_2d"], dtype=np.float32),
            confidence=float(data["confidence"]),
            timestamp=float(data["timestamp"]),
            mano_params=mano
        )


class HandTrackerABC(ABC):
    """Abstract Base Class for 3D Hand Pose and Mesh Estimator (e.g. Fast-HaMeR / MediaPipe / Mock)."""

    @abstractmethod
    def estimate(
        self,
        image: np.ndarray,
        intrinsics: Optional[np.ndarray] = None
    ) -> List[HandPose]:
        """
        Estimate 3D/2D hand poses from an RGB image frame.
        
        Args:
            image: RGB image frame of shape (H, W, 3) as np.uint8.
            intrinsics: 3x3 camera intrinsic matrix K.
            
        Returns:
            List of detected HandPose objects.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state or temporal filters."""
        pass
