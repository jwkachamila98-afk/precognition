"""Unit and integration tests for Phase 4 Discrepancy Engine and Residual Adaptation Policy."""

import numpy as np
import pytest
from src.perception.hand_tracker import HandPose, HandSide
from src.perception.scene_parser import BoundingBox3D
from src.simulation.trajectory_generator import ForeseenWaypoint
from src.policy.discrepancy import DiscrepancyEngine, DiscrepancyState
from src.mocks.mock_policy import MockResidualPolicy
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.transport.ws_server import WSInferenceServer
from src.transport.ws_client import WSStreamingClient


def test_discrepancy_engine_112d_state():
    engine = DiscrepancyEngine()

    hand_tracker = MockHandTracker()
    real_hand = hand_tracker.estimate(np.zeros((480, 640, 3), dtype=np.uint8))[0]

    bbox = BoundingBox3D(
        label="remote_control",
        center=np.array([0.08, 0.12, 0.58], dtype=np.float32),
        size=np.array([0.06, 0.18, 0.03], dtype=np.float32)
    )

    foreseen_wp = ForeseenWaypoint(
        timestep=15,
        time_offset=0.5,
        hand_keypoints_3d=real_hand.keypoints_3d + 0.02, # Small 2cm tracking error
        hand_keypoints_2d=real_hand.keypoints_2d,
        wrist_pose=np.array([0.08, 0.08, 0.48, 0.0, 0.0, 0.0], dtype=np.float32),
        object_pose=np.array([0.08, 0.12, 0.58, 0.0, 0.0, 0.0], dtype=np.float32),
        contact_state=np.zeros(5, dtype=np.float32),
        gripper_aperture=0.0
    )

    last_action = np.full(7, 0.01, dtype=np.float32)

    disc_state = engine.evaluate(
        real_hand=real_hand,
        foreseen_step=foreseen_wp,
        target_object=bbox,
        last_action=last_action
    )

    assert isinstance(disc_state, DiscrepancyState)
    assert disc_state.state_vector.shape == (112,)
    assert -1.0 <= disc_state.reward <= 1.0
    assert disc_state.discrepancy_norm >= 0.0
    assert disc_state.pose_error >= 0.0


def test_mock_residual_policy_evaluation():
    policy = MockResidualPolicy(state_dim=112, action_dim=7, max_residual=0.08)
    dummy_state = np.random.uniform(-1.0, 1.0, size=(112,)).astype(np.float32)

    action = policy.evaluate(dummy_state)
    assert action.joint_residuals.shape == (7,)
    assert np.all(action.joint_residuals >= -0.0801)
    assert np.all(action.joint_residuals <= 0.0801)
    assert 0.0 <= action.gripper_action <= 1.0


def test_mock_residual_policy_online_adaptation():
    policy = MockResidualPolicy(state_dim=112, action_dim=7)
    policy.reset()
    assert policy.step_count == 0

    # Simulate 35 online transitions to trigger PPO update at step 30
    for i in range(35):
        s = np.random.randn(112).astype(np.float32)
        a = policy.evaluate(s).joint_residuals
        r = 0.5 + 0.01 * i # Improving reward
        policy.record_transition(state=s, action=a, reward=r)

    assert policy.step_count == 35
    assert len(policy.buffer) == 35
    assert len(policy.loss_history) >= 2
    assert policy.cumulative_adaptations >= 1


@pytest.mark.asyncio
async def test_phase4_client_server_e2e():
    port = 8796
    server = WSInferenceServer(
        host="127.0.0.1",
        port=port,
        hand_tracker=MockHandTracker(),
        depth_estimator=MockDepthEstimator(),
        scene_parser=MockSceneParser(),
        affordance_extractor=MockAffordanceExtractor(),
        trajectory_diffusion=MockTrajectoryDiffusion(),
        discrepancy_engine=DiscrepancyEngine(),
        physics_engine=MockPhysicsEngine(),
        policy=MockResidualPolicy()
    )

    await server.start()
    client = WSStreamingClient(host="127.0.0.1", port=port)

    try:
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        response = await client.send_frame(frame, frame_id=1, intent="foresee me picking this remote control")

        assert response is not None
        assert response.frame_id == 1
        assert response.policy_residuals is not None
        assert len(response.policy_residuals) == 7
        assert -1.0 <= response.reward_score <= 1.0
        assert response.discrepancy_norm >= 0.0
        assert response.adaptation_status in ("ACTIVE", "PAUSED")
        assert response.buffer_step_count >= 1
    finally:
        await client.close()
        await server.stop()
