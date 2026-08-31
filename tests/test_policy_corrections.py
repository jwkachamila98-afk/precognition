"""Visible policy corrections on the plan (tests/test_policy_corrections.py).

The arrows are a claim about the network: that learning has moved the plan,
by this much, in this direction. So the things worth pinning are the ones
that would turn the claim into decoration - drawing arrows when nothing has
been learned, and drawing them at a magnitude the plan does not actually
carry.
"""

import types

import numpy as np
import pytest

from apps.local_client import INK, LocalVisualizer
from src.mocks.mock_trajectory_diffusion import minimum_jerk_step
from src.perception.scene_parser import BoundingBox3D
from src.simulation.trajectory_generator import ForeseenTrajectory, ForeseenWaypoint


def _vis():
    """A visualizer with just the configuration the drawing code reads."""
    vis = LocalVisualizer.__new__(LocalVisualizer)
    vis.config = types.SimpleNamespace(
        visualization=types.SimpleNamespace(draw_skeleton=True,
                                            draw_bounding_box=True,
                                            draw_depth_inset=False))
    vis.draw_scale = 1.0
    vis._panel_cache = {}
    return vis


def _plan(bias, n=60):
    """A rollout shaped like the generator's: the bias is fully present from
    the end of the approach phase (28%) onward, not ramped to the end."""
    start = np.array([-0.16, 0.14, 0.50], dtype=np.float32)
    grasp = np.array([0.06, 0.02, 0.60], dtype=np.float32) + bias
    lift = grasp + np.array([0.0, -0.14, 0.02], dtype=np.float32)
    wps = []
    for i in range(n):
        t = i / (n - 1)
        if t <= 0.28:
            p = start + minimum_jerk_step(t / 0.28) * (grasp - start)
        elif t <= 0.50:
            p = grasp.copy()
        else:
            p = grasp + minimum_jerk_step((t - 0.50) / 0.50) * (lift - grasp)
        wps.append(ForeseenWaypoint(
            timestep=i + 1, time_offset=i / 30.0,
            hand_keypoints_3d=np.zeros((21, 3), np.float32),
            hand_keypoints_2d=np.zeros((21, 2), np.float32),
            wrist_pose=np.array([p[0], p[1], p[2], 0, 0, 0], np.float32),
            object_pose=np.zeros(6, np.float32),
            contact_state=np.zeros(5, np.float32)))
    return ForeseenTrajectory(intent="pick up the cup", target_label="cup",
                              waypoints=wps)


def _frame():
    return np.full((480, 640, 3), 60, np.uint8)


@pytest.mark.parametrize("bias", [
    None,
    np.zeros(3, dtype=np.float32),
    np.array([0.001, -0.0005, 0.0], dtype=np.float32),   # under a millimetre
])
def test_nothing_is_drawn_before_anything_has_been_learned(bias):
    """An arrow on screen must always mean the network changed something. A
    baseline policy that has learned nothing yet has nothing to point at."""
    vis, frame = _vis(), _frame()
    before = frame.copy()
    vis.draw_policy_corrections(frame, _plan(np.zeros(3, np.float32)), bias)
    assert np.array_equal(frame, before), "corrections drawn with no learning"


def test_a_real_bias_draws_the_correction():
    vis, frame = _vis(), _frame()
    before = frame.copy()
    bias = np.array([0.022, -0.013, 0.0], dtype=np.float32)
    vis.draw_policy_corrections(frame, _plan(bias), bias)
    assert not np.array_equal(frame, before), "a learned bias drew nothing"


def test_an_absent_plan_is_not_an_error():
    """Detection drops out constantly; there is often no plan to annotate."""
    vis, frame = _vis(), _frame()
    before = frame.copy()
    bias = np.array([0.02, 0.0, 0.0], dtype=np.float32)
    vis.draw_policy_corrections(frame, None, bias)
    vis.draw_policy_corrections(
        frame, ForeseenTrajectory(intent="", target_label="", waypoints=[]), bias)
    assert np.array_equal(frame, before)


def _held_plan(bias, n=60):
    """A rollout that holds one fully-corrected pose for its whole length.

    Every sampled waypoint then projects to the SAME head point, so the accent
    pixels bound a single arrow and their extent is that arrow's length. On a
    plan that moves, the three arrows sit at three different places along the
    path and their combined bounding box says nothing about how long any one
    of them is - which is how an earlier version of this test passed against
    deliberately wrong arrow scaling.
    """
    p = np.array([0.06, 0.02, 0.60], dtype=np.float32) + bias
    return ForeseenTrajectory(
        intent="pick up the cup", target_label="cup",
        waypoints=[ForeseenWaypoint(
            timestep=i + 1, time_offset=i / 30.0,
            hand_keypoints_3d=np.zeros((21, 3), np.float32),
            hand_keypoints_2d=np.zeros((21, 2), np.float32),
            wrist_pose=np.array([p[0], p[1], p[2], 0, 0, 0], np.float32),
            object_pose=np.zeros(6, np.float32),
            contact_state=np.zeros(5, np.float32)) for i in range(n)])


def test_the_arrow_matches_the_correction_the_plan_actually_carries():
    """The arrow's length is the claim being made about the network.

    The generator folds the bias into the grasp point and everything after it,
    reaching full correction at the end of the approach phase - so over the
    stretch these arrows are drawn on, the plan carries the WHOLE bias. An
    earlier version scaled them linearly with the waypoint index and drew
    roughly half the real correction, understating the thing being
    demonstrated.
    """
    # A purely horizontal correction, so the arrow lies in one row band and
    # can be measured clear of the caption - which is drawn in the same accent
    # colour, is far wider than the arrow, and otherwise dominates any
    # measurement made over the whole frame.
    bias = np.array([0.030, 0.0, 0.0], dtype=np.float32)
    h, w = 480, 640
    fx = 0.8 * w

    def proj(p):
        z = max(float(p[2]), 0.1)
        return np.array([fx * p[0] / z + w / 2.0, fx * p[1] / z + h / 2.0])

    plan = _held_plan(bias)
    wrist = np.asarray(plan.waypoints[0].wrist_pose[:3], dtype=np.float32)
    head = proj(wrist)
    expected = float(np.linalg.norm(head - proj(wrist - bias)))

    vis, frame = _vis(), _frame()
    vis.draw_policy_corrections(frame, plan, bias)

    # The dotted plan line is a quiet neutral, so accent pixels in the arrow's
    # own rows are the arrow alone - and on a held plan all three coincide.
    band = frame[int(head[1]) - 5:int(head[1]) + 6]
    cols = np.argwhere(np.all(band == np.array(INK["blue"]), axis=2))[:, 1]
    assert len(cols) > 0, "no accent-coloured correction was drawn"
    span = float(cols.max() - cols.min())
    assert span >= 0.85 * expected, (
        f"arrow spans {span:.0f} px for a {expected:.0f} px correction - "
        "the drawn correction understates the plan's real nudge")
