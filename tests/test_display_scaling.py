"""Annotations drawn at card resolution (tests/test_display_scaling.py).

Annotations used to be drawn on the 640x480 feed and scaled up threefold into
the video card, leaving every line and label soft beside the crisp chrome. They
are now drawn at the card's own size - but the landmarks feeding them are the
same objects being recorded and scored, so the scaling must never touch them.
"""

import types

import numpy as np
import pytest

from apps.local_client import LocalClientRunner
from src.perception.hand_tracker import HandPose, HandSide
from src.ui.stage import Stage


def _runner(stage):
    """Just enough of the runner to exercise the display-scaling helpers."""
    r = LocalClientRunner.__new__(LocalClientRunner)
    r.stage = stage
    r.visualizer = types.SimpleNamespace(draw_scale=1.0)
    return r


def _pose(seed=0):
    rng = np.random.default_rng(seed)
    return HandPose(
        hand_id=0, side=HandSide.RIGHT,
        keypoints_3d=rng.normal(0, 0.04, (21, 3)).astype(np.float32),
        keypoints_2d=rng.uniform(0, 640, (21, 2)).astype(np.float32),
        confidence=0.9, timestamp=0.0)


def test_frame_and_landmarks_are_enlarged_together():
    stage = Stage(1920, 804)
    runner = _runner(stage)
    frame = np.zeros((480, 640, 3), np.uint8)
    poses = [_pose()]
    original = poses[0].keypoints_2d.copy()

    out, display = runner._to_display_resolution(frame, poses)

    vx1, vy1, vx2, vy2 = stage.layout.video
    assert out.shape[1] == vx2 - vx1 and out.shape[0] == vy2 - vy1

    sx = (vx2 - vx1) / 640.0
    sy = (vy2 - vy1) / 480.0
    assert np.allclose(display[0].keypoints_2d[:, 0], original[:, 0] * sx, atol=1e-3)
    assert np.allclose(display[0].keypoints_2d[:, 1], original[:, 1] * sy, atol=1e-3)
    assert runner.visualizer.draw_scale == pytest.approx((sx + sy) / 2, abs=1e-3)


def test_the_recorded_poses_are_never_rescaled():
    """These same objects are recorded and scored. Rescaling them in place would
    silently corrupt the episode the user is performing."""
    stage = Stage(1920, 804)
    runner = _runner(stage)
    poses = [_pose(1), _pose(2)]
    before = [p.keypoints_2d.copy() for p in poses]

    _, display = runner._to_display_resolution(np.zeros((480, 640, 3), np.uint8), poses)

    for pose, original in zip(poses, before):
        assert np.array_equal(pose.keypoints_2d, original), "a recorded pose was mutated"
    assert display[0] is not poses[0], "display poses must be copies"
    assert not np.array_equal(display[0].keypoints_2d, poses[0].keypoints_2d)


def test_ghost_replay_poses_are_scaled_to_match_the_card():
    """The ghost is replayed from recordings in sensor pixels; unscaled it would
    be drawn at a fraction of its correct position on the enlarged card."""
    runner = _runner(Stage(1920, 804))
    runner.visualizer.draw_scale = 1.38
    poses = [_pose(3)]
    before = poses[0].keypoints_2d.copy()

    scaled = runner._scale_poses_for_display(poses)
    assert np.allclose(scaled[0].keypoints_2d, before * 1.38, atol=1e-3)
    assert np.array_equal(poses[0].keypoints_2d, before), "originals must survive"


def test_scaling_is_a_no_op_when_the_card_matches_the_sensor():
    runner = _runner(Stage(1920, 804))
    runner.visualizer.draw_scale = 1.0
    poses = [_pose(4)]
    assert runner._scale_poses_for_display(poses) is poses


def test_the_execution_phase_explains_itself():
    """USER_EXECUTING advances on frames in which a hand was DETECTED, not on
    elapsed time, so it stalls indefinitely and silently when tracking drops.
    That ended three live sessions before an episode ever completed, so the
    count and the reason for a stall are both on screen."""
    from src.policy.workflow_state import ExecutionPhase

    _, body = LocalClientRunner._status_message(
        ExecutionPhase.USER_EXECUTING, "coffee cup", False, progress=0.4, tracking=True)
    assert "24/60" in body, f"frame progress not shown: {body}"

    _, stalled = LocalClientRunner._status_message(
        ExecutionPhase.USER_EXECUTING, "coffee cup", False, progress=0.4, tracking=False)
    assert "Raise your hand" in stalled and "nothing is being captured" in stalled, \
        f"a stall must say why: {stalled}"

    # Every phase must produce something to show.
    for phase in ExecutionPhase:
        title, text = LocalClientRunner._status_message(phase, "cup", True)
        assert title and text, f"{phase} has no guidance"


def test_the_render_loop_actually_invokes_every_annotation_renderer():
    """A wiring guard, not a rendering one.

    Inserting the display-resize step once REPLACED the draw_hand_skeleton call
    rather than preceding it, and the skeleton silently stopped being drawn. The
    entire suite still passed: every test covered the drawing functions in
    isolation, and nothing asserted the loop still called them.
    """
    import inspect
    from apps import local_client

    source = inspect.getsource(local_client.LocalClientRunner.run)
    for renderer in ("draw_hand_skeleton", "draw_3d_bounding_boxes",
                     "draw_affordance_hotspots", "draw_hand_replay",
                     "_update_lab_panel", "_compose_stage"):
        assert f"{renderer}(" in source, f"the render loop no longer calls {renderer}"

    # And the annotations must be fed the DISPLAY poses, not the sensor-space
    # originals, or they land at a fraction of their correct position.
    skeleton_call = source[source.index("draw_hand_skeleton("):][:200]
    assert "display_poses" in skeleton_call, "skeleton is not using display-scaled poses"
