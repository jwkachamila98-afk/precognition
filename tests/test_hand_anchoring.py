"""Where a tracked hand actually is (tests/test_hand_anchoring.py).

MediaPipe returns metric landmarks relative to the hand's own geometric centre.
The tracker used to copy those through and paste a constant 0.5 m into z, which
pinned every hand to the optical axis and discarded its position in the scene.

Nothing caught it, because the mock tracker - which every other test uses -
places its wrist at a genuine camera-frame position and projects it properly.
The fixtures were healthier than the real sensor path.
"""

import numpy as np
import pytest

from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.perception.hand_tracker import HandPose, HandSide
from src.perception.mediapipe_tracker import MediaPipeHandTracker
from src.perception.scene_parser import BoundingBox3D
from src.policy.discrepancy import DiscrepancyEngine

WIDTH, HEIGHT, FOV = 640, 480, 60.0


def _tracker():
    """The anchoring helper without standing up MediaPipe's inference graph."""
    t = MediaPipeHandTracker.__new__(MediaPipeHandTracker)
    t.horizontal_fov_deg = FOV
    return t


def _focal():
    return (0.5 * WIDTH) / np.tan(np.radians(FOV) * 0.5)


def _hand_at(position, seed=0):
    """A plausible metric hand placed at a true camera-frame position."""
    rng = np.random.default_rng(seed)
    shape = rng.normal(0, 0.035, (21, 3)).astype(np.float32)
    return (shape - shape.mean(axis=0)) + np.asarray(position, np.float32)


def _as_mediapipe_gives_it(world):
    """Project to pixels, then centre the metric landmarks - what the sensor hands back."""
    f = _focal()
    px = np.stack([world[:, 0] * f / world[:, 2] + WIDTH * 0.5,
                   world[:, 1] * f / world[:, 2] + HEIGHT * 0.5], axis=1).astype(np.float32)
    return (world - world.mean(axis=0)).astype(np.float32), px


@pytest.mark.parametrize("position", [
    (0.0, 0.0, 0.50), (0.20, 0.05, 0.55), (-0.30, 0.12, 0.70),
    (0.40, -0.10, 0.60), (0.10, 0.02, 0.35), (-0.15, -0.20, 0.95),
])
def test_a_hand_is_recovered_where_it_actually_was(position):
    """Centred landmarks plus their projection determine the hand's position."""
    world = _hand_at(position)
    local, px = _as_mediapipe_gives_it(world)
    got = _tracker()._anchor_in_camera_frame(local, px, WIDTH, HEIGHT)
    assert np.linalg.norm(got.mean(axis=0) - np.asarray(position)) < 0.01
    # The shape must survive the move intact, not just the centroid.
    assert np.allclose(got - got.mean(axis=0), local, atol=2e-3)


def test_hands_at_different_image_positions_do_not_collapse_together():
    """The specific regression: every hand used to land on the optical axis."""
    t = _tracker()
    left = t._anchor_in_camera_frame(*_as_mediapipe_gives_it(_hand_at((-0.25, 0.0, 0.6))),
                                     width=WIDTH, height=HEIGHT)
    right = t._anchor_in_camera_frame(*_as_mediapipe_gives_it(_hand_at((0.25, 0.0, 0.6))),
                                      width=WIDTH, height=HEIGHT)
    separation = float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))
    assert separation > 0.40, f"two hands 50 cm apart came back {separation*100:.1f} cm apart"


def test_a_degenerate_hand_falls_back_without_crashing():
    """Too foreshortened or too small to localise must not produce nonsense."""
    t = _tracker()
    for local, px in [
        (np.zeros((21, 3), np.float32), np.full((21, 2), 320.0, np.float32)),
        (_hand_at((0, 0, 0.5)) * 0.0, np.zeros((21, 2), np.float32)),
        (np.full((21, 3), 1e-7, np.float32), np.full((21, 2), 5.0, np.float32)),
    ]:
        out = t._anchor_in_camera_frame(local, px, WIDTH, HEIGHT)
        assert out.shape == (21, 3)
        assert np.all(np.isfinite(out))
        assert 0.05 < float(out[:, 2].mean()) < 2.0, "fell back to an impossible distance"


def _reward_for(object_centre, execution_offset):
    """Score a run where the user tracks the plan, displaced by a fixed amount."""
    bbox = BoundingBox3D(label="cup", center=np.asarray(object_centre, np.float32),
                         size=np.array([0.09, 0.09, 0.09], np.float32))
    traj = MockTrajectoryDiffusion().generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox, num_steps=60,
        affordance_map=MockAffordanceExtractor().extract_affordance(bbox, "pick up the cup"))
    plan = np.stack([w.hand_keypoints_3d for w in traj.waypoints])
    t = _tracker()
    real = np.stack([t._anchor_in_camera_frame(*_as_mediapipe_gives_it(p + np.asarray(execution_offset)),
                                               width=WIDTH, height=HEIGHT) for p in plan])
    poses = [HandPose(hand_id=0, side=HandSide.RIGHT, keypoints_3d=k.astype(np.float32),
                      keypoints_2d=np.zeros((21, 2), np.float32), confidence=1.0,
                      timestamp=i / 30.0) for i, k in enumerate(real)]
    return DiscrepancyEngine().compile_episode_discrepancy(traj, poses, policy=None)


def test_reward_measures_the_attempt_not_where_the_object_sits():
    """The bug that pinned episode reward at -1.000.

    With the hand stuck on the optical axis, the error against a plan authored
    at the object was dominated by the object's offset - so a perfectly executed
    reach scored as a total failure whenever the object was off-centre, and
    every episode looked identical to the policy.
    """
    rewards = [_reward_for(c, (0.0, 0.0, 0.0)).episode_reward
               for c in [(0.02, 0.03, 0.50), (0.30, 0.0, 0.55), (0.50, -0.15, 0.75)]]
    assert min(rewards) > 0.9, f"a perfect reach should score high everywhere, got {rewards}"
    assert max(rewards) - min(rewards) < 0.05, "score drifted with the object's position"


def test_reward_still_separates_a_good_attempt_from_a_bad_one():
    """Invariance must not have been bought by making the reward constant."""
    centre = (0.30, 0.0, 0.55)
    perfect = _reward_for(centre, (0.0, 0.0, 0.0)).episode_reward
    sloppy = _reward_for(centre, (0.08, 0.0, 0.0)).episode_reward
    bad = _reward_for(centre, (0.25, 0.0, 0.0)).episode_reward
    assert perfect > sloppy > bad, f"not ordered: {perfect:.3f} {sloppy:.3f} {bad:.3f}"
    assert perfect - bad > 0.8, "reward has too little range to learn from"


def test_the_lab_replay_finds_the_real_moment_of_grasp():
    """A consequence of anchoring, and the reason the reenactment used to grasp
    at the wrong moment.

    The lab picks its contact frame as the one where the fingertips come closest
    to the object in 3-D. With every hand pinned to the optical axis that
    comparison was meaningless - on this recording it chose frame 3 of 90, before
    the reach had begun - so the ghost hand closed on nothing and the object
    began moving far too early.
    """
    from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
    from src.simulation.lab_sim import LabSimulator

    gen = MockTrajectoryDiffusion()
    centre = np.array([0.30, 0.0, 0.55], np.float32)
    frames, grasp_at = 90, 60

    # A reach that settles into a real grasp pose - the wrist stops short of the
    # object, as a hand actually does - holds, then lifts away.
    grasp_wrist = gen._solve_grasp_wrist(centre, np.array([-0.28, -0.18, 0.0], np.float32))
    world = []
    for i in range(frames):
        if i <= grasp_at:
            approach = 1.0 - i / float(grasp_at)
            wrist = grasp_wrist + np.array([-0.28, -0.20, -0.10], np.float32) * approach
        else:
            lift = (i - grasp_at) / float(frames - grasp_at)
            wrist = grasp_wrist + np.array([0.0, -0.14, 0.02], np.float32) * lift
        world.append(gen._generate_hand_keypoints_3d(
            wrist.astype(np.float32), gen._rot_grasp, 0.15 if i < grasp_at else 0.55))

    t = _tracker()
    poses = [HandPose(hand_id=0, side=HandSide.RIGHT,
                      keypoints_3d=t._anchor_in_camera_frame(*_as_mediapipe_gives_it(k),
                                                             width=WIDTH, height=HEIGHT),
                      keypoints_2d=np.zeros((21, 2), np.float32), confidence=1.0,
                      timestamp=1788038400.0 + i / 30.0)
             for i, k in enumerate(world)]

    sim = LabSimulator(width=96, height=72)
    bbox = BoundingBox3D(label="cup", center=centre,
                         size=np.array([0.09, 0.09, 0.09], np.float32))
    assert sim.prepare_from_demonstration(poses, bbox, None)
    assert abs(sim._contact_step() - grasp_at) <= 6, \
        f"grasp detected at frame {sim._contact_step()}, actually at {grasp_at}"


def test_both_tracker_backends_share_one_anchoring_implementation():
    """There are two MediaPipe backends. The anchoring was written twice, the
    server used the copy that was NOT fixed, and the whole repair silently did
    nothing in production - the logs showed MediaPipeTasksHandTracker while the
    fix sat in MediaPipeHandTracker. One implementation, imported by both.
    """
    import inspect
    from src.perception import hand_anchoring, mediapipe_tasks_hand_tracker
    from src.perception import mediapipe_tracker as mp_tracker

    for module in (mp_tracker, mediapipe_tasks_hand_tracker):
        source = inspect.getsource(module)
        assert "anchor_hand" in source, f"{module.__name__} does not anchor its hands"
        # The exact shape of the original bug: a constant pasted into depth.
        assert "0.50 + wlm.z" not in source, \
            f"{module.__name__} still pins world landmarks to a fixed depth"

    assert mp_tracker.anchor_hand is hand_anchoring.anchor_hand
    assert mediapipe_tasks_hand_tracker.anchor_hand is hand_anchoring.anchor_hand


def test_the_anchored_hand_lives_in_the_same_camera_as_everything_else():
    """One pinhole, or the hand and the object never share a geometry.

    Every other metres-to-pixels step in this system uses fx = 0.8 * width
    (BoundingBox3D.project_to_2d, the affordance hotspots, the trajectory
    generator, the lab camera). This module used to assume a 60-degree
    horizontal field of view instead, which is fx = 0.866 * width. Both numbers
    are guesses about an uncalibrated webcam; the requirement is only that they
    be the SAME guess.

    They were not, and the cost was not cosmetic: the recovered depth came back
    8% long - 45 mm of 3-D error at a normal reach - so a hand visually touching
    an object still carried an offset. Episode reward, the co-adaptation wrist
    bias and the lab's grasp-frame detection are all computed from exactly that
    relationship, and the learned bias the demo exists to show is itself only a
    couple of centimetres. The error was larger than the signal.
    """
    import cv2

    from src.perception.hand_anchoring import anchor_hand

    W, H = 640, 480
    fx = 0.8 * W
    rng = np.random.default_rng(3)

    local = rng.normal(0, 0.030, (21, 3)).astype(np.float32)
    local -= local.mean(axis=0)

    # A known pose: off-centre and rotated, since an on-axis hand would hide a
    # focal-length disagreement almost entirely.
    R, _ = cv2.Rodrigues(np.array([0.3, -0.5, 0.2]))
    t = np.array([0.10, -0.06, 0.55])
    truth = local @ R.T + t
    uv = np.stack([fx * truth[:, 0] / truth[:, 2] + W / 2,
                   fx * truth[:, 1] / truth[:, 2] + H / 2], 1).astype(np.float32)

    recovered = anchor_hand(local, uv, W, H)

    assert np.abs(recovered - truth).max() < 1e-3, (
        "the anchored hand does not land where it was observed - the tracker "
        "and the projection convention disagree about the camera again")
