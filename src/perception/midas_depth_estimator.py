"""Real monocular metric-ish depth estimation using Intel ISL's MiDaS.

MiDaS predicts relative inverse depth (disparity) from a single RGB frame using an
actual trained neural network, rather than a synthetic procedural depth field. Absolute
scale is not observable from a monocular image, so the disparity map is mapped into
the configured [min_depth, max_depth] band; relative structure (what's near vs far,
object silhouettes) is real, derived from the actual scene.

DISPARITY IS NOT DEPTH. It is 1/depth, and the two are related reciprocally,
so the band is interpolated in DISPARITY and inverted at the end. Interpolating
straight into metres - which this did - bends every plane in the scene: a desk
receding from the camera came back curved by 42 cm even given a perfect depth
range, and a scene reconstructed from it read as a lumpy relief rather than a
room.

SCALE IS NOT METRIC, BUT IT IS STABLE. The band is fitted to a carried,
rate-limited percentile range rather than to each frame's own extremes.
Per-frame normalisation re-scaled the entire scene whenever anything entered
or left view - the same bottle measured 24 cm in one reenactment and 59 cm in
the next - which corrupts anything that compares two frames, object extent and
the learned wrist bias included.
"""

import logging
import time
from typing import Optional

import numpy as np

from src.perception.depth_estimator import DepthEstimatorABC, DepthMap

logger = logging.getLogger(__name__)


class MiDaSDepthEstimator(DepthEstimatorABC):
    """Real single-image depth estimator, best available MiDaS variant, GPU when possible."""

    # How fast the carried disparity range follows the current frame. Slow
    # enough that a hand crossing the foreground does not rescale the room,
    # fast enough to follow the camera being moved to a new scene.
    _RANGE_EMA = 0.05
    # Hard ceiling on how far the range may move in one frame, as a
    # fraction of its own value.
    _MAX_RANGE_STEP = 0.05

    # Best first. MiDaS_small is midas_v21_small_256 - a 2021 mobile-grade
    # network taking a 256 px input - and it was the only thing ever loaded
    # here, on a machine with a 4090 idling at 2% while it ran. Its output is
    # smooth enough that a desk reconstructs as a relief rather than a plane,
    # which no amount of correct disparity handling downstream can undo.
    #
    # DPT_Large is a few hundred milliseconds on CPU and tens of milliseconds
    # on this GPU, so on the pod it is close to free; the smaller variants are
    # here for a laptop, which is also the only place the cost would show.
    _MODELS = (
        ("DPT_Large", "dpt_transform"),
        ("DPT_Hybrid", "dpt_transform"),
        ("MiDaS_small", "small_transform"),
    )

    def __init__(self, min_depth: float = 0.15, max_depth: float = 1.8,
                 model_name: Optional[str] = None) -> None:
        import cv2
        import torch

        self.min_depth = min_depth
        self.max_depth = max_depth
        # Carried disparity range, so the scene's scale does not jump between
        # frames. None until the first frame sets it.
        self._range = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cv2 = cv2
        self._torch = torch

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        candidates = ([(model_name, "dpt_transform")] if model_name
                      else list(self._MODELS))
        last_error: Optional[Exception] = None
        for name, transform_attr in candidates:
            try:
                self.model = torch.hub.load("intel-isl/MiDaS", name)
                self.model.to(self.device).eval()
                self.transform = getattr(transforms, transform_attr)
                self.model_name = name
                logger.info(f"MiDaSDepthEstimator: loaded {name} on "
                            f"device={self.device}")
                return
            except Exception as exc:                # a weight download can fail
                last_error = exc
                logger.warning(f"MiDaSDepthEstimator: could not load {name} "
                               f"({exc}); trying the next one.")
        raise RuntimeError(f"no MiDaS model could be loaded: {last_error}")

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
        d_min, d_max = self._normalisation_range(disparity)
        norm = (disparity - d_min) / (d_max - d_min + 1e-6)  # 0 = far, 1 = near
        norm = np.clip(norm, 0.0, 1.0)

        # Disparity is INVERSE depth, so it has to be inverted, not remapped.
        # Interpolating linearly from disparity into metres - which is what this
        # did - bends every flat surface in the scene: a desk receding from the
        # camera came back curved by 42 cm even given a perfect depth range, and
        # a reconstruction built from it reads as a lumpy relief rather than a
        # room. Interpolating in DISPARITY space and inverting at the end is the
        # same one-line cost and recovers a plane exactly.
        inv_near, inv_far = 1.0 / self.min_depth, 1.0 / self.max_depth
        depth_m = (1.0 / (norm * (inv_near - inv_far) + inv_far)).astype(np.float32)

        return DepthMap(
            depth=depth_m,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            timestamp=time.time(),
            intrinsics=None,
        )

    def _normalisation_range(self, disparity: np.ndarray) -> tuple:
        """The disparity range to map into the depth band, eased across frames.

        Normalising to each frame's own extremes re-scales the whole scene
        whenever anything enters or leaves view: the same water bottle measured
        24 cm in one reenactment and 59 cm in the next, three minutes apart,
        because a hand had moved through the foreground in between and pulled
        the disparity maximum with it. Every metre in the scene moved with it.

        Percentiles rather than min/max, so one speck of noise at either end
        cannot set the scale, and an exponential carry so the range walks
        instead of jumping. It does not make the depth metric - a single image
        cannot - but it makes it STABLE, which is what anything comparing two
        frames actually depends on.
        """
        lo = float(np.percentile(disparity, 1.0))
        hi = float(np.percentile(disparity, 99.0))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        if self._range is None:
            self._range = (lo, hi)
            return self._range

        def follow(prev: float, target: float) -> float:
            # Eased, then rate-limited RELATIVE to the value itself. The easing
            # alone is not enough: a hand filling a corner of the frame pushes
            # the disparity maximum up sixfold, and five percent of a sixfold
            # jump is still a quarter of the scene's scale in one frame. The
            # limit bounds the lurch however extreme the intruder, while still
            # letting the range migrate over a second or so when the camera
            # genuinely moves somewhere new.
            step = self._RANGE_EMA * (target - prev)
            cap = self._MAX_RANGE_STEP * max(abs(prev), 1e-6)
            return prev + float(np.clip(step, -cap, cap))

        prev_lo, prev_hi = self._range
        self._range = (follow(prev_lo, lo), follow(prev_hi, hi))
        return self._range

    def reset(self) -> None:
        self._range = None
