"""Unit and integration tests for Phase 8 Robot Hardware, Policy Checkpointing, and Co-Adaptation Analytics."""

import time
from pathlib import Path
import numpy as np
import pytest

from src.hardware.robot_interface import MockRobotHardware, ROS2ControlBridge, RobotState
from src.policy.checkpointing import PolicyCheckpointManager
from src.analytics.benchmark import CoAdaptationBenchmark, TrialMetrics
from src.policy.discrepancy import DiscrepancyEngine, EpisodeDiscrepancyReport
from src.mocks.mock_policy import MockResidualPolicy
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.perception.intent_parser import MockLLMIntentParser
from src.transport.ws_server import WSInferenceServer
from src.transport.ws_client import WSStreamingClient


def test_mock_robot_hardware():
    robot = MockRobotHardware(dof=7)
    assert robot.is_connected

    # Test joint command
    target_q = np.array([0.5, -0.2, 0.1, -1.0, 0.0, 1.2, 0.0], dtype=np.float32)
    success = robot.send_joint_commands(target_q, gripper_command=0.8)
    assert success

    state = robot.read_joint_states()
    assert isinstance(state, RobotState)
    assert state.joint_positions.shape == (7,)
    assert state.joint_velocities.shape == (7,)
    assert state.gripper_aperture == 0.8
    assert not state.is_e_stopped

    # Test Emergency Stop
    robot.emergency_stop()
    state_estop = robot.read_joint_states()
    assert state_estop.is_e_stopped
    assert not robot.send_joint_commands(target_q)

    # Test Reset
    robot.reset_e_stop()
    assert not robot.read_joint_states().is_e_stopped


def test_ros2_control_bridge_fallback():
    bridge = ROS2ControlBridge()
    assert bridge.is_connected
    success = bridge.send_joint_commands(np.zeros(7), gripper_command=0.5)
    assert success
    state = bridge.read_joint_states()
    assert state.gripper_aperture == 0.5


def test_policy_checkpoint_manager(tmp_path: Path):
    manager = PolicyCheckpointManager(base_profiles_dir=str(tmp_path / "profiles"))
    policy = MockResidualPolicy()

    # Modify policy weights
    policy.W[0, 0] = 0.42
    policy.b[0] = -0.15
    policy.step_count = 35

    # Save checkpoint
    saved_path = manager.save_checkpoint(policy, user_id="user_alpha")
    assert saved_path.exists()
    assert (tmp_path / "profiles" / "user_alpha" / "latest.json").exists()

    # Reset to baseline
    manager.reset_to_baseline(policy)
    assert policy.W[0, 0] == 0.0
    assert policy.step_count == 0

    # Restore checkpoint
    loaded = manager.load_checkpoint(policy, user_id="user_alpha")
    assert loaded
    assert abs(policy.W[0, 0] - 0.42) < 1e-4
    assert policy.step_count == 35

    # List checkpoints
    ckpts = manager.list_checkpoints(user_id="user_alpha")
    assert len(ckpts) >= 1


def test_coadaptation_benchmark(tmp_path: Path):
    benchmark = CoAdaptationBenchmark(log_dir=str(tmp_path / "benchmarks"))
    assert benchmark.total_trials == 0

    # Trial 1 (High initial error)
    rep_1 = EpisodeDiscrepancyReport(
        mean_pose_error=0.080, # 80mm
        max_pose_error=0.120,
        smoothness_variance=0.04,
        contact_misalignment=0.05,
        episode_reward=0.20,
        num_steps_sim=60,
        num_steps_real=60
    )
    benchmark.record_trial(rep_1, intent="remote control")

    # Trial 2 (Improved error)
    rep_2 = EpisodeDiscrepancyReport(
        mean_pose_error=0.040, # 40mm
        max_pose_error=0.060,
        smoothness_variance=0.02,
        contact_misalignment=0.02,
        episode_reward=0.75,
        num_steps_sim=60,
        num_steps_real=60
    )
    benchmark.record_trial(rep_2, intent="remote control")

    assert benchmark.total_trials == 2
    reduction = benchmark.compute_error_reduction_pct()
    assert abs(reduction - 50.0) < 1.0 # 50% error reduction

    summary = benchmark.get_summary()
    assert summary["total_trials"] == 2
    assert summary["initial_error_mm"] == 80.0
    assert summary["latest_error_mm"] == 40.0

    # Test ASCII chart output
    chart = benchmark.format_ascii_trend_graph()
    assert "CO-ADAPTATION MULTI-TRIAL LEARNING CONVERGENCE" in chart
    assert "OVERALL ERROR REDUCTION" in chart

    # Test JSON and CSV export
    json_path = benchmark.export_summary_json()
    assert json_path.exists()
    csv_path = benchmark.export_csv()
    assert csv_path.exists()


@pytest.mark.asyncio
async def test_phase8_websocket_hardware_e2e():
    port = 8794
    server = WSInferenceServer(
        host="127.0.0.1",
        port=port,
        hand_tracker=MockHandTracker(),
        depth_estimator=MockDepthEstimator(),
        intent_parser=MockLLMIntentParser(),
        scene_parser=MockSceneParser(),
        affordance_extractor=MockAffordanceExtractor(),
        trajectory_diffusion=MockTrajectoryDiffusion(),
        discrepancy_engine=DiscrepancyEngine(),
        physics_engine=MockPhysicsEngine(),
        policy=MockResidualPolicy(),
        robot=MockRobotHardware(dof=7)
    )

    await server.start()
    client = WSStreamingClient(host="127.0.0.1", port=port)

    try:
        frame = np.full((480, 640, 3), 110, dtype=np.uint8)
        response = await client.send_frame(
            frame,
            frame_id=1,
            intent="grasp the red coffee cup by the handle"
        )

        assert response is not None
        assert response.robot_state is not None
        assert "joint_positions" in response.robot_state
        assert len(response.robot_state["joint_positions"]) == 7
        assert response.benchmark_summary is not None
        assert "total_trials" in response.benchmark_summary
    finally:
        await client.close()
        await server.stop()
