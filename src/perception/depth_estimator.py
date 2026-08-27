"""Perception interfaces and dataclasses for monocular/metric depth estimation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DepthMap:
    """Metric depth map representation."""
    # 2D depth array in meters (float32, shape: H x W)
    depth: np.ndarray
    min_depth: float
    max_depth: float
    timestamp: float
    intrinsics: Optional[np.ndarray] = None # 3x3 camera matrix

    @property
    def shape(self) -> tuple:
        return self.depth.shape

    def to_colored_heatmap(self, colormap: int = 2) -> np.ndarray:
        """
        Convert depth to an 8-bit BGR color heatmap for visualization.
        Default colormap is cv2.COLORMAP_JET (2) or cv2.COLORMAP_INFERNO.
        """
        import cv2
        clipped = np.clip(self.depth, self.min_depth, self.max_depth)
        normalized = ((clipped - self.min_depth) / (self.max_depth - self.min_depth + 1e-6) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, colormap)
        return colored


class DepthEstimatorABC(ABC):
    """Abstract Base Class for Metric Depth Estimators (e.g. Depth Anything V2, ZoeDepth, Mock)."""

    @abstractmethod
    def estimate_depth(self, image: np.ndarray) -> DepthMap:
        """
        Estimate metric depth from an RGB image frame.

        Args:
            image: RGB image frame (H, W, 3) as np.uint8.

        Returns:
            DepthMap containing per-pixel metric depth in meters.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state if temporal smoothing is used."""
        pass
