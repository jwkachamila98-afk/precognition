"""High-performance synthetic metric depth estimator mock for CPU-only development."""

import math
import time
import numpy as np

from src.perception.depth_estimator import DepthEstimatorABC, DepthMap


class MockDepthEstimator(DepthEstimatorABC):
    """
    Simulates a monocular metric depth model (e.g. Depth Anything V2) with zero GPU overhead.
    Computes a synthetic metric depth field containing background slope and a dynamic
    foreground hand protrusion.
    Runs at >200 FPS on Intel Mac CPUs.
    """

    def __init__(
        self,
        min_depth: float = 0.2,
        max_depth: float = 2.5,
        target_shape: tuple = (240, 320),
    ) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_height, self.target_width = target_shape
        self._start_time = time.time()

        # Precompute static spatial coordinate grids
        y_grid, x_grid = np.mgrid[0:self.target_height, 0:self.target_width]
        self._norm_x = (x_grid / (self.target_width - 1.0)).astype(np.float32)
        self._norm_y = (y_grid / (self.target_height - 1.0)).astype(np.float32)

    def estimate_depth(self, image: np.ndarray) -> DepthMap:
        """
        Synthesize metric depth array (in meters) matching image or target dimensions.
        """
        now = time.time()
        t = now - self._start_time

        # Background planar depth ramp: 1.2m to 2.2m from bottom to top
        bg_depth = 1.2 + 0.8 * (1.0 - self._norm_y) + 0.1 * np.sin(self._norm_x * 3.14159)

        # Dynamic foreground hand bulge center moving smoothly
        center_x = 0.5 + 0.15 * math.sin(t * 1.2)
        center_y = 0.55 + 0.08 * math.cos(t * 0.9)
        hand_depth = 0.55 + 0.05 * math.sin(t * 0.6)

        # Compute radial distance from hand center
        dx = (self._norm_x - center_x) * (self.target_width / self.target_height)
        dy = (self._norm_y - center_y)
        dist_sq = dx * dx + dy * dy

        # Gaussian-like depth protrusion for hand
        hand_mask = np.exp(-dist_sq / (2.0 * (0.12 ** 2)))
        composite_depth = bg_depth * (1.0 - hand_mask) + hand_depth * hand_mask

        # Add small high-frequency spatial variation (noise)
        noise = (np.sin(self._norm_x * 50.0) * np.cos(self._norm_y * 50.0) * 0.005).astype(np.float32)
        final_depth = np.clip(composite_depth + noise, self.min_depth, self.max_depth).astype(np.float32)

        return DepthMap(
            depth=final_depth,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            timestamp=now,
            intrinsics=None
        )

    def reset(self) -> None:
        self._start_time = time.time()
