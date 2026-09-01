"""Turning MiDaS disparity into depth (tests/test_depth_conversion.py).

MiDaS predicts DISPARITY - inverse depth - up to an unknown scale and shift.
Absolute metres are not recoverable from one image and this does not pretend
otherwise. What it must get right is the SHAPE: a flat surface has to come back
flat, and the scale must not lurch between frames.

The estimator itself needs torch, a GPU and a downloaded model, so the
conversion is exercised through the same arithmetic the class performs rather
than by standing MiDaS up.
"""

import numpy as np
import pytest

MIN_D, MAX_D = 0.15, 1.8


def _to_depth(disparity, lo, hi):
    """The conversion as MiDaSDepthEstimator.estimate_depth performs it."""
    norm = np.clip((disparity - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    inv_near, inv_far = 1.0 / MIN_D, 1.0 / MAX_D
    return 1.0 / (norm * (inv_near - inv_far) + inv_far)


def _bend_cm(z):
    """How far a surface deviates from flat, after the best scale and offset -
    shape error only, since the absolute scale is unknowable anyway."""
    x = np.linspace(0.0, 1.0, len(z))
    A = np.stack([x, np.ones_like(x)], axis=1)
    fit = A @ np.linalg.lstsq(A, z, rcond=None)[0]
    return float(np.abs(z - fit).max()) * 100.0


def test_a_flat_surface_comes_back_flat():
    """The bug that made the reconstructed room read as a lumpy relief.

    Depth was interpolated LINEARLY from disparity into metres. Disparity is
    1/z, so that bends every plane in the scene - a desk receding from 0.5 m to
    2.0 m came back curved by 42 cm even when the assumed depth range was
    exactly right. Interpolating in disparity and inverting at the end is the
    same arithmetic cost and recovers the plane exactly.
    """
    z_true = np.linspace(MIN_D, MAX_D, 40)
    disparity = 1.0 / z_true                    # what the network predicts

    recovered = _to_depth(disparity, disparity.min(), disparity.max())
    assert _bend_cm(recovered) < 0.5, (
        f"a flat surface came back bent by {_bend_cm(recovered):.1f} cm")

    # And the old mapping really did bend it, so this test is not vacuous.
    norm = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    old = MAX_D - norm * (MAX_D - MIN_D)
    assert _bend_cm(old) > 10.0, "the linear mapping was not actually the problem"


def test_depth_is_monotonic_in_disparity():
    """Nearer disparity must always mean nearer depth. An inversion that got a
    sign wrong would turn the room inside out while still looking plausible in
    a single still."""
    disparity = np.linspace(0.2, 5.0, 50)
    depth = _to_depth(disparity, disparity.min(), disparity.max())
    assert np.all(np.diff(depth) < 0), "larger disparity did not mean nearer"
    assert depth.min() == pytest.approx(MIN_D, rel=1e-3)
    assert depth.max() == pytest.approx(MAX_D, rel=1e-3)


def test_the_carried_range_does_not_lurch_when_the_foreground_changes():
    """The same water bottle measured 24 cm in one reenactment and 59 cm in the
    next, because normalising to each frame's own extremes rescales the whole
    scene whenever anything enters or leaves view.

    Exercises the estimator's range logic without constructing it, since that
    would need torch and a model download.
    """
    from src.perception.midas_depth_estimator import MiDaSDepthEstimator

    est = MiDaSDepthEstimator.__new__(MiDaSDepthEstimator)
    est._range = None

    room = np.random.default_rng(0).uniform(0.5, 2.0, (64, 64))
    steady = [MiDaSDepthEstimator._normalisation_range(est, room) for _ in range(30)][-1]

    # A hand crosses the foreground: a big, close disparity spike over 5% of frame.
    intruded = room.copy()
    intruded[:14, :14] = 12.0
    after = MiDaSDepthEstimator._normalisation_range(est, intruded)

    drift = abs(after[1] - steady[1]) / max(steady[1], 1e-6)
    assert drift < 0.10, (
        f"one frame with a hand in it moved the scale by {drift * 100:.0f}%")
