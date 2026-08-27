"""Unit and integration tests for Phase 3 components."""

import numpy as np
import pytest
from src.perception.hand_tracker import HandPose, HandSide
from src.perception.scene_parser import BoundingBox3D
from src.simulation.simulator import SimAction
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_policy import MockPolicy
from src.transport.ws_server import WSInferenceServer
from src.transport.ws_client import WSStreamingClient


def test_mock_affordance_extractor():
    extractor = MockAffordanceExtractor(num_surface_points=100)
    bbox = BoundingBox3D(
        label="remote_control",
        center=np.array([0.08, 0.12, 0.58], dtype=np.float32),
        size=np.array([0.06, 0.18, 0.03], dtype=np.float32)
    )

    affordance = extractor.extract_affordance(bbox, intent="foresee me picking this remote control")
    assert affordance.object_label == "remote_control"
    assert affordance.surface_points.shape == (100, 3)
    assert affordance.contact_probabilities.shape == (100,)
    assert np.all(affordance.contact_probabilities >= 0.0)
    assert np.all(affordance.contact_probabilities <= 1.0)
    assert len(affordance.hotspots) >= 2


def test_mock_trajectory_diffusion():
    generator = MockTrajectoryDiffusion()
    bbox = BoundingBox3D(
        label="remote_control",
        center=np.array([0.08, 0.12, 0.58], dtype=np.float32),
        size=np.array([0.06, 0.18, 0.03], dtype=np.float32)
    )
    extractor = MockAffordanceExtractor(num_surface_points=50)
    affordance = extractor.extract_affordance(bbox)

    hand_tracker = MockHandTracker()
    start_pose = hand_tracker.estimate(np.zeros((480, 640, 3), dtype=np.uint8))[0]

    trajectory = generator.generate_foreseen_rollout(
        start_hand_pose=start_pose,
        target_object=bbox,
        affordance_map=affordance,
        intent="foresee me picking this remote control",
        num_steps=60
    )

    assert trajectory.num_waypoints == 60
    assert trajectory.intent == "foresee me picking this remote control"
    assert trajectory.target_label == "remote_control"

    # Verify waypoint 1 (start) vs waypoint 60 (lifted end)
    wp_start = trajectory.waypoints[0]
    wp_end = trajectory.waypoints[-1]

    assert wp_start.timestep == 1
    assert wp_end.timestep == 60
    assert wp_start.hand_keypoints_3d.shape == (21, 3)
    assert wp_start.hand_keypoints_2d.shape == (21, 2)
    assert wp_end.contact_state.shape == (5,)
    assert wp_start.gripper_aperture <= 0.2
    assert wp_end.gripper_aperture >= 0.8


def test_mock_physics_engine():
    engine = MockPhysicsEngine(num_dof=7)
    obj_id = engine.instantiate_object_mesh(
        mesh_name="remote_control",
        position=np.array([0.08, 0.08, 0.48], dtype=np.float32)
    )
    assert obj_id == 1

    state = engine.reset()
    assert state.joint_positions.shape == (7,)

    # Step with grasp command
    action = SimAction(
        target_joint_positions=np.zeros(7, dtype=np.float32),
        gripper_command=0.9
    )
    new_state = engine.step(action)
    assert new_state.contact_forces.shape == (6,)
    assert new_state.contact_forces[2] > 0.0 # Contact force detected


@pytest.mark.asyncio
async def test_phase3_websocket_e2e():
    port = 8797
    server = WSInferenceServer(
        host="127.0.0.1",
        port=port,
        hand_tracker=MockHandTracker(),
        depth_estimator=MockDepthEstimator(),
        scene_parser=MockSceneParser(),
        affordance_extractor=MockAffordanceExtractor(),
        trajectory_diffusion=MockTrajectoryDiffusion(),
        physics_engine=MockPhysicsEngine(),
        policy=MockPolicy()
    )

    await server.start()
    client = WSStreamingClient(host="127.0.0.1", port=port)

    try:
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        response = await client.send_frame(frame, frame_id=1, intent="foresee me picking this remote control")

        assert response is not None
        assert response.frame_id == 1
        assert response.foreseen_trajectory is not None
        assert response.affordance_map is not None

        foreseen = response.get_foreseen_trajectory()
        assert foreseen is not None
        assert foreseen.num_waypoints == 60

        affordance = response.get_affordance_map()
        assert affordance is not None
        assert len(affordance.hotspots) >= 2
    finally:
        await client.close()
        await server.stop()
