"""The trial-to-trial adaptation loop (tests/test_coadaptation_loop.py).

The planner shifts its grasp waypoint by a learned bias, and each episode
reports how far the user's hand ended up from that plan. Whether the loop
actually converges depends on what is measured and how it is fed back - and it
used to do neither correctly.
"""

import numpy as np
import pytest

from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.perception.hand_tracker import HandPose, HandSide
from src.perception.scene_parser import BoundingBox3D
from src.policy.discrepancy import DiscrepancyEngine, EpisodeDiscrepancyReport

BIAS_MAX, ALPHA = 0.05, 0.4


def _fixture(personal_offset):
    """A user who reaches for the cup their own way, the same way every trial."""
    gen, ex = MockTrajectoryDiffusion(), MockAffordanceExtractor()
    bbox = BoundingBox3D(label="cup", center=np.array([0.30, 0.0, 0.55], np.float32),
                         size=np.array([0.09, 0.09, 0.09], np.float32))
    aff = ex.extract_affordance(bbox, "pick up the cup")
    base = gen.generate_foreseen_rollout(start_hand_pose=None, target_object=bbox,
                                         affordance_map=aff, num_steps=60)
    world = (np.stack([w.hand_keypoints_3d for w in base.waypoints])
             + np.asarray(personal_offset, np.float32))
    poses = [HandPose(hand_id=0, side=HandSide.RIGHT, keypoints_3d=k.astype(np.float32),
                      keypoints_2d=np.zeros((21, 2), np.float32), confidence=1.0,
                      timestamp=i / 30.0) for i, k in enumerate(world)]
    return gen, bbox, aff, poses


def _episode(gen, bbox, aff, poses, bias):
    traj = gen.generate_foreseen_rollout(start_hand_pose=None, target_object=bbox,
                                         affordance_map=aff, num_steps=60,
                                         learned_bias=np.asarray(bias, np.float32))
    return DiscrepancyEngine().compile_episode_discrepancy(traj, poses, policy=None)


def test_the_grasp_offset_cancels_one_for_one_with_the_applied_bias():
    """The property that makes an integral update the right one: the residual
    measured over the closing approach falls by exactly what is applied, so it
    reaches zero when the bias equals the user's real offset."""
    personal = -0.06
    gen, bbox, aff, poses = _fixture((personal, 0.0, 0.0))
    measured = [_episode(gen, bbox, aff, poses, (b, 0.0, 0.0)).grasp_wrist_offset[0]
                for b in (0.0, -0.02, -0.04, -0.06)]
    assert measured[0] == pytest.approx(personal, abs=0.005)
    for applied, residual in zip((0.0, -0.02, -0.04, -0.06), measured):
        assert residual == pytest.approx(personal - applied, abs=0.005)
    assert measured[-1] == pytest.approx(0.0, abs=0.005), "a full correction must cancel it"


def _converge(rule, field, personal, trials=8):
    gen, bbox, aff, poses = _fixture(personal)
    bias = np.zeros(3, np.float32)
    errors = []
    for _ in range(trials):
        rep = _episode(gen, bbox, aff, poses, bias)
        errors.append(rep.mean_pose_error)
        step = np.clip(np.asarray(getattr(rep, field), np.float32), -BIAS_MAX, BIAS_MAX)
        bias = np.clip((1 - ALPHA) * bias + ALPHA * step if rule == "proportional"
                       else bias + ALPHA * step, -BIAS_MAX, BIAS_MAX)
    return errors, bias


def test_adaptation_converges_on_how_this_user_actually_reaches():
    """Eight trials should very nearly eliminate a consistent personal offset."""
    personal = (-0.04, -0.02, 0.0)
    errors, bias = _converge("integral", "grasp_wrist_offset", personal)
    assert errors[-1] < errors[0] * 0.30, f"error barely moved: {errors[0]:.4f} -> {errors[-1]:.4f}"
    assert errors[-1] < 0.015, f"{errors[-1]*100:.2f} cm of avoidable error left standing"
    assert np.allclose(bias[:2], np.asarray(personal[:2]), atol=0.008), \
        f"recovered {bias[:2]} for a user offset of {personal[:2]}"


def test_averaging_toward_the_residual_leaves_permanent_error():
    """Why the update rule had to change.

    episode_offset is what remains AFTER the current bias was applied. Blending
    the bias toward it treats a residual as a total, which is a proportional
    controller - it settles where the correction it produces exactly sustains
    the error that produced it, and never gets closer.
    """
    personal = (-0.04, -0.02, 0.0)
    prop_errors, prop_bias = _converge("proportional", "mean_wrist_offset", personal)
    int_errors, _ = _converge("integral", "grasp_wrist_offset", personal)

    assert prop_errors[-1] > 2.0 * int_errors[-1], "the two rules should differ clearly"
    # It stalls around half the user's real offset rather than reaching it.
    assert abs(prop_bias[0]) < 0.75 * abs(personal[0])
    settled = prop_errors[-3:]
    assert max(settled) - min(settled) < 0.002, "should be stuck, not still improving"


def test_a_report_without_a_grasp_offset_still_adapts():
    """Checkpoints written before the grasp window existed must keep working."""
    legacy = {"mean_pose_error": 0.05, "max_pose_error": 0.1, "smoothness_variance": 0.0,
              "contact_misalignment": 0.02, "episode_reward": 0.3, "num_steps_sim": 60,
              "num_steps_real": 55, "policy_loss_delta": 0.0,
              "mean_wrist_offset": [-0.03, 0.01, 0.0]}
    rep = EpisodeDiscrepancyReport.from_dict(legacy)
    assert rep.grasp_wrist_offset == [-0.03, 0.01, 0.0], "should fall back to the episode mean"
    assert EpisodeDiscrepancyReport.from_dict(rep.to_dict()).grasp_wrist_offset == \
        rep.grasp_wrist_offset, "round trip must preserve it"
