"""Real monocular metric-ish depth estimation using Intel ISL's MiDaS (small variant).

MiDaS predicts relative inverse depth (disparity) from a single RGB frame using an
actual trained neural network, rather than a synthetic procedural depth field. Absolute
scale is not observable from a monocular image, so the disparity map is rescaled into
the configured [min_depth, max_depth] band; relative structure (what's near vs far,
object silhouettes) is real, derived from the actual scene.
"""

import logging
import time

import numpy as np

from src.perception.depth_estimator import DepthEstimatorABC, DepthMap

logger = logging.getLogger(__name__)


class MiDaSDepthEstimator(DepthEstimatorABC):
    """Real single-image depth estimator (MiDaS_small) running on GPU when available."""

    def __init__(self, min_depth: float = 0.15, max_depth: float = 1.8) -> None:
        import cv2
        import torch

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cv2 = cv2
        self._torch = torch

        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform
        logger.info(f"MiDaSDepthEstimator: loaded MiDaS_small on device={self.device}")

    def estimate_depth(self, image: np.ndarray) -> DepthMap:
        torch = self._torch
        rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        disparity = prediction.cpu().numpy()
        d_min, d_max = float(disparity.min()), float(disparity.max())
        norm = (disparity - d_min) / (d_max - d_min + 1e-6)  # 0 = far, 1 = near
        depth_m = (self.max_depth - norm * (self.max_depth - self.min_depth)).astype(np.float32)

        return DepthMap(
            depth=depth_m,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            timestamp=time.time(),
            intrinsics=None,
        )

    def reset(self) -> None:
        pass
