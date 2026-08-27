"""Scene point cloud, 3D bounding primitive, and geometric spatial parser interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from src.perception.depth_estimator import DepthMap


@dataclass
class BoundingBox3D:
    """Oriented 3D bounding box primitive for an object in camera coordinates."""
    label: str
    center: np.ndarray # (3,) [x, y, z] in meters
    size: np.ndarray # (3,) [extent_x, extent_y, extent_z] in meters
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32)) # [roll, pitch, yaw]
    confidence: float = 1.0

    @property
    def corners_3d(self) -> np.ndarray:
        """Compute the 8 3D corner coordinates of the bounding box (8, 3)."""
        dx, dy, dz = self.size / 2.0
        # Canonical unit box corners
        corners = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz]
        ], dtype=np.float32)

        # Simple Euler rotation (pitch, yaw, roll)
        rx, ry, rz = self.rotation
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        rotated = (R @ corners.T).T
        return rotated + self.center

    def project_to_2d(self, intrinsics: Optional[np.ndarray] = None, image_shape: tuple = (480, 640)) -> np.ndarray:
        """Project 8 3D corners to 2D image coordinates (8, 2)."""
        h, w = image_shape[:2]
        if intrinsics is None:
            fx = fy = 0.8 * w
            cx = w / 2.0
            cy = h / 2.0
        else:
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]

        corners = self.corners_3d
        corners_2d = np.zeros((8, 2), dtype=np.float32)
        valid_z = np.clip(corners[:, 2], 0.1, 10.0)
        corners_2d[:, 0] = fx * (corners[:, 0] / valid_z) + cx
        corners_2d[:, 1] = fy * (corners[:, 1] / valid_z) + cy
        return corners_2d

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "rotation": self.rotation.tolist(),
            "confidence": float(self.confidence)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox3D":
        return cls(
            label=data["label"],
            center=np.array(data["center"], dtype=np.float32),
            size=np.array(data["size"], dtype=np.float32),
            rotation=np.array(data.get("rotation", [0.0, 0.0, 0.0]), dtype=np.float32),
            confidence=float(data.get("confidence", 1.0))
        )


@dataclass
class ScenePointCloud:
    """3D point cloud representation of the perceived physical scene."""
    # (N, 3) XYZ coordinates in camera optical frame
    points: np.ndarray
    # (N, 3) RGB color normalized [0, 1] or [0, 255]
    colors: Optional[np.ndarray] = None
    # (N, 3) surface normals
    normals: Optional[np.ndarray] = None
    timestamp: float = 0.0

    @property
    def num_points(self) -> int:
        return len(self.points)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": self.points.tolist(),
            "colors": self.colors.tolist() if self.colors is not None else None,
            "normals": self.normals.tolist() if self.normals is not None else None,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenePointCloud":
        pts = np.array(data["points"], dtype=np.float32)
        cols = np.array(data["colors"], dtype=np.float32) if data.get("colors") is not None else None
        norms = np.array(data["normals"], dtype=np.float32) if data.get("normals") is not None else None
        return cls(points=pts, colors=cols, normals=norms, timestamp=data.get("timestamp", 0.0))


@dataclass
class ParsedScene:
    """Combined parsed spatial scene with point cloud and object bounding primitives."""
    intent: str
    bounding_boxes: List[BoundingBox3D]
    point_cloud: Optional[ScenePointCloud] = None
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "bounding_boxes": [b.to_dict() for b in self.bounding_boxes],
            "point_cloud": self.point_cloud.to_dict() if self.point_cloud is not None else None,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedScene":
        bboxes = [BoundingBox3D.from_dict(b) for b in data.get("bounding_boxes", [])]
        pcd = ScenePointCloud.from_dict(data["point_cloud"]) if data.get("point_cloud") is not None else None
        return cls(
            intent=data.get("intent", ""),
            bounding_boxes=bboxes,
            point_cloud=pcd,
            timestamp=data.get("timestamp", 0.0)
        )


class SceneParserABC(ABC):
    """Abstract Base Class for reconstructing and parsing the 3D scene from RGB-D and intent."""

    @abstractmethod
    def parse_scene(
        self,
        image: np.ndarray,
        depth: DepthMap,
        intent: str = "foresee me picking this remote control",
        intrinsics: Optional[np.ndarray] = None
    ) -> ParsedScene:
        """
        Backproject 2D RGB-D data and segment target object primitives conditioned on user intent.

        Args:
            image: RGB image frame (H, W, 3)
            depth: DepthMap containing metric depth
            intent: Natural language task intent
            intrinsics: 3x3 camera calibration matrix

        Returns:
            ParsedScene containing bounding primitives and point cloud.
        """
        pass
