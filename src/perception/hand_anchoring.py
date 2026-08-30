"""Placing tracked hands in the camera frame (src/perception/hand_anchoring.py).

MediaPipe reports its metric hand landmarks relative to the hand's own geometric
centre. Where the hand IS in the scene has to be recovered separately, and both
tracker backends need it, so the recovery lives here rather than being written
twice - it was already implemented twice, once per backend, and only one copy
got fixed.

Everything downstream that compares a tracked hand against a plan authored at an
object depends on this: episode reward, the co-adaptation offset, and the
simulated lab's choice of when the grasp happened.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Outside this band the size-based estimate is not believable: nearer and the
# hand is touching the lens, further and it is out of the workspace.
MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 1.60
NOMINAL_DEPTH_M = 0.50

# Most laptop webcams sit near 60 degrees horizontally. An error here scales the
# recovered depth proportionally but does not distort the hand's shape.
DEFAULT_HFOV_DEG = 60.0


def focal_px(width: int, hfov_deg: float = DEFAULT_HFOV_DEG) -> float:
    return (0.5 * float(width)) / np.tan(np.radians(hfov_deg) * 0.5)


def anchor_hand(
    local: np.ndarray,
    kpts_2d: np.ndarray,
    width: int,
    height: int,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> np.ndarray:
    """Place hand-centred metric landmarks at their true camera-frame position.

    This is a pose-from-correspondences problem: the metric shape is known, its
    projection is observed, and the rigid transform between them is wanted - so
    it is solved as one. A simpler apparent-size estimate (Z = f x metres /
    pixels) was tried first and is biased, because it assumes the measured span
    lies on the optical axis; off to one side it over-estimated depth by up to
    20 cm at 40 cm off-centre, which is precisely the regime that matters.

    `local` must be the metric landmarks with their centroid at the origin.
    """
    f = focal_px(width, hfov_deg)
    K = np.array([[f, 0.0, width * 0.5],
                  [0.0, f, height * 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    ok, rvec, tvec = False, None, None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            local.astype(np.float64), kpts_2d.astype(np.float64), K, None,
            flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        ok = False

    if ok and tvec is not None:
        depth = float(np.asarray(tvec).reshape(-1)[2])
        if MIN_DEPTH_M <= depth <= MAX_DEPTH_M:
            R, _ = cv2.Rodrigues(rvec)
            return (local.astype(np.float64) @ R.T
                    + np.asarray(tvec).reshape(1, 3)).astype(np.float32)

    # No usable solution - too foreshortened, too small, or too blurred to
    # localise. Sit the hand at the nominal working distance on the ray through
    # its own centre, which is still better than pinning it to the optical axis.
    centre_px = np.asarray(kpts_2d, dtype=np.float64).mean(axis=0)
    out = np.asarray(local, dtype=np.float64).copy()
    out[:, 0] += (centre_px[0] - width * 0.5) * NOMINAL_DEPTH_M / f
    out[:, 1] += (centre_px[1] - height * 0.5) * NOMINAL_DEPTH_M / f
    out[:, 2] += NOMINAL_DEPTH_M
    return out.astype(np.float32)
