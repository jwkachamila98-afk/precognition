"""Spoken actions become motion (tests/test_action_plans.py).

`action_type` used to be parsed and then read by nobody, so every utterance -
"push it", "point at it", "pick it up" - produced the same reach and lift. The
schema describes an action along axes the trajectory generator can execute,
which is what lets a phrase nobody anticipated still produce sensible motion.
"""

import numpy as np
import pytest

from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.perception.action_schema import (APPROACHES, CONTACTS, FOLLOW_THROUGHS,
                                          ActionPlan, plan_from_text)
from src.perception.scene_parser import BoundingBox3D


def _rollout(action):
    gen, ex = MockTrajectoryDiffusion(), MockAffordanceExtractor()
    bbox = BoundingBox3D(label="cup", center=np.array([0.10, 0.0, 0.55], np.float32),
                         size=np.array([0.08, 0.10, 0.08], np.float32))
    traj = gen.generate_foreseen_rollout(
        start_hand_pose=None, target_object=bbox,
        affordance_map=ex.extract_affordance(bbox, "cup"), num_steps=60, action=action)
    wrists = np.stack([w.wrist_pose[:3] for w in traj.waypoints])
    objects = np.stack([w.object_pose[:3] for w in traj.waypoints])
    return wrists, objects, traj


def test_passing_no_action_preserves_the_original_pick_and_lift():
    """The tuned pick-up must not shift underneath everything that depends on it."""
    a_w, a_o, _ = _rollout(None)
    b_w, b_o, _ = _rollout(None)
    assert np.array_equal(a_w, b_w) and np.array_equal(a_o, b_o)
    assert a_o[-1][1] < a_o[0][1] - 0.10, "the object should end up lifted (-Y is up)"


def test_different_verbs_produce_different_motion():
    base, _, _ = _rollout(None)
    seen = {}
    for utterance in ("push the cup aside", "point at the cup",
                      "drink from the cup", "hand me the cup"):
        wrists, _, _ = _rollout(plan_from_text(utterance))
        divergence = float(np.abs(wrists - base).max())
        assert divergence > 0.05, f"'{utterance}' moved like a plain pick-up"
        seen[utterance] = wrists
    # And they differ from EACH OTHER, not merely from the default.
    keys = list(seen)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert float(np.abs(seen[a] - seen[b]).max()) > 0.02, f"{a} == {b}"


def test_only_a_hand_that_closed_on_the_object_carries_it():
    """Pointing at a cup used to drag the cup through the air."""
    _, pointed, _ = _rollout(plan_from_text("point at the cup"))
    assert float(np.linalg.norm(pointed[-1] - pointed[0])) < 0.005

    _, grasped, _ = _rollout(plan_from_text("pick up the cup"))
    assert float(np.linalg.norm(grasped[-1] - grasped[0])) > 0.10

    _, pushed, _ = _rollout(plan_from_text("push the cup aside"))
    assert float(np.linalg.norm(pushed[-1] - pushed[0])) > 0.05, "a push should move it"


def test_a_push_slides_along_the_surface_rather_than_lifting():
    _, pushed, _ = _rollout(plan_from_text("push the cup out of the way"))
    travel = pushed[-1] - pushed[0]
    assert abs(travel[0]) > abs(travel[1]) * 2.0, f"a push should be lateral, got {travel}"


def test_handing_over_brings_the_object_toward_the_camera():
    _, handed, _ = _rollout(plan_from_text("hand me the cup"))
    assert handed[-1][2] < handed[0][2] - 0.05, "should travel toward the viewer (-Z)"


def test_a_tilt_rotates_the_wrist():
    _, _, traj = _rollout(plan_from_text("drink from the cup"))
    roll = np.array([w.wrist_pose[5] for w in traj.waypoints])
    assert abs(float(roll[-1] - roll[0])) > np.radians(20), "no wrist roll for a tilt"


@pytest.mark.parametrize("utterance", [
    "tip the last bit out of that", "nudge it towards me", "check the lid",
    "", "aslkdjh qwerty", "please could you possibly do the thing",
])
def test_any_phrasing_yields_a_plan_the_generator_can_execute(utterance):
    """An unrecognised verb must still produce motion, not an exception."""
    plan = plan_from_text(utterance)
    assert plan.approach in APPROACHES
    assert plan.contact in CONTACTS
    assert plan.follow_through in FOLLOW_THROUGHS
    wrists, _, _ = _rollout(plan)
    assert np.all(np.isfinite(wrists))


def test_impossible_combinations_are_reconciled():
    """A model will sometimes describe carrying something it never grasped."""
    plan = ActionPlan(contact="none", follow_through="lift", grip=0.9,
                      travel_m=5.0, tilt_deg=999).validated()
    assert plan.follow_through != "lift", "nothing is carried by a hand that never closed"
    assert plan.grip <= 0.25
    assert 0.0 <= plan.travel_m <= 0.45
    assert -120 <= plan.tilt_deg <= 120


def test_a_plan_survives_a_round_trip():
    plan = plan_from_text("hand me the spoon")
    assert ActionPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()
