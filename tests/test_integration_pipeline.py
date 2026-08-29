"""End-to-end integration and stress testing suite across all system components (Phase 9)."""

import time
from pathlib import Path
import numpy as np
import pytest

from src.perception.hand_tracker import HandPose, HandSide
from src.perception.intent_parser import MockLLMIntentParser, ParsedIntent
from src.perception.scene_parser import BoundingBox3D
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.mocks.mock_policy import MockResidualPolicy
from src.policy.discrepancy import DiscrepancyEngine, EpisodeDiscrepancyReport
from src.policy.workflow_state import ExecutionPhase, WorkflowController
from src.policy.checkpointing import PolicyCheckpointManager
from src.analytics.benchmark import CoAdaptationBenchmark
from src.hardware.robot_interface import MockRobotHardware
from src.safety.safety_monitor import SafetyMonitor, SafetyStatus
from src.transport.ws_server import WSInferenceServer
from src.transport.ws_client import WSStreamingClient


def test_full_visuomotor_pipeline_end_to_end():
    """
    Test complete visuomotor pipeline execution path:
    Voice intent -> Structured LLM parser -> 3D scene parsing -> Affordance extraction ->
    60-step diffusion rollout -> Discrepancy compilation -> Policy adaptation -> Safety filtering -> Robot execution.
    """
    # 1. Voice Intent Parsing
    intent_parser = MockLLMIntentParser()
    raw_speech = "grasp the red coffee cup by the handle"
    parsed_intent = intent_parser.parse_intent(raw_speech)
    assert parsed_intent.target_object == "coffee cup"
    assert parsed_intent.affordance_hotspot == "handle"

    # 2. Perception: Scene Parsing & Depth
    scene_parser = MockSceneParser()
    depth_estimator = MockDepthEstimator()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_map = depth_estimator.estimate_depth(dummy_img)
    scene = scene_parser.parse_scene(dummy_img, depth_map, intent=parsed_intent.target_object)
    assert len(scene.bounding_boxes) == 1
    target_box = scene.bounding_boxes[0]

    # 3. Affordance & 60-Step Trajectory Diffusion
    affordance_extractor = MockAffordanceExtractor()
    affordance_map = affordance_extractor.extract_affordance(target_box, intent=raw_speech)
    assert len(affordance_map.hotspots) >= 1

    diffusion = MockTrajectoryDiffusion()
    foreseen_traj = diffusion.generate_foreseen_rollout(
        start_hand_pose=None,
        target_object=target_box,
        affordance_map=affordance_map,
        intent=raw_speech,
        num_steps=60
    )
    assert len(foreseen_traj.waypoints) == 60

    # 4. Discrepancy Engine & Episode Discrepancy Compilation
    discrepancy_engine = DiscrepancyEngine()
    policy = MockResidualPolicy()
    
    # Simulate real physical execution (60 frames)
    real_poses = []
    for t in range(60):
        kpts = np.zeros((21, 3), dtype=np.float32)
        kpts[:, 2] = 0.50 - 0.002 * t + 0.005 * np.random.randn()
        p = HandPose(
            hand_id=0,
            side=HandSide.RIGHT,
            keypoints_3d=kpts,
            keypoints_2d=np.zeros((21, 2), dtype=np.float32),
            confidence=0.95,
            timestamp=time.time()
        )
        real_poses.append(p)

    report = discrepancy_engine.compile_episode_discrepancy(foreseen_traj, real_poses, policy=policy)
    assert isinstance(report, EpisodeDiscrepancyReport)
    assert report.mean_pose_error >= 0.0
    assert -1.0 <= report.episode_reward <= 1.0

    # 5. Multi-Trial Co-Adaptation Benchmark
    benchmark = CoAdaptationBenchmark()
    metric = benchmark.record_trial(report, intent=raw_speech)
    assert benchmark.total_trials == 1
    assert metric.mean_pose_error == report.mean_pose_error

    # 6. Safety Guardrail & Robot Command Dispatch
    safety_monitor = SafetyMonitor(dof=7)
    robot = MockRobotHardware(dof=7)
    robot.connect()

    action = policy.evaluate(np.zeros(112, dtype=np.float32))
    target_q = np.zeros(7, dtype=np.float32)
    target_q[:len(action.joint_residuals)] = action.joint_residuals

    safety = safety_monitor.evaluate_safety(
        target_q=target_q,
        current_q=robot.read_joint_states().joint_positions,
        cartesian_pos=np.array([0.05, 0.10, 0.50], dtype=np.float32),
        last_packet_time=time.time(),
        obstacles=scene.bounding_boxes,
        dt=0.033
    )
    assert safety.is_safe
    assert len(safety.clamped_joint_positions) == 7

    success = robot.send_joint_commands(safety.clamped_joint_positions, gripper_command=action.gripper_action)
    assert success

    robot_state = robot.read_joint_states()
    assert robot_state.is_connected
    assert not robot_state.is_e_stopped


def test_safety_monitor_kinematic_and_velocity_clamping():
    """Verify SafetyMonitor clamps extreme joint positions and excessive velocities."""
    safety = SafetyMonitor(dof=7, max_velocity=2.0)
    current_q = np.zeros(7, dtype=np.float32)

    # 1. Test excessive position command (10.0 rad > joint limit 2.89 rad)
    excessive_q = np.full(7, 10.0, dtype=np.float32)
    clamped_q, clamped_qd, warnings = safety.filter_joint_command(excessive_q, current_q, dt=0.033)
    
    assert "JOINT_LIMIT_REACHED" in warnings
    assert np.all(clamped_q <= safety.upper_limits + 1e-4)

    # 2. Test velocity saturation (|qd| <= 2.0 rad/s)
    # Attempt step from 0.0 to 1.0 in dt=0.01s (requires 100 rad/s)
    step_target = np.full(7, 1.0, dtype=np.float32)
    clamped_q_v, clamped_qd_v, warnings_v = safety.filter_joint_command(step_target, current_q, dt=0.01)
    
    assert "VELOCITY_SATURATED" in warnings_v
    assert np.all(np.abs(clamped_qd_v) <= 2.0 + 1e-4)


def test_safety_monitor_heartbeat_timeout():
    """Verify SafetyMonitor triggers emergency stop upon telemetry loss > 250ms."""
    safety = SafetyMonitor(dof=7, heartbeat_timeout_sec=0.250)
    current_q = np.zeros(7, dtype=np.float32)
    target_q = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Simulating packet from 0.400s ago (> 250ms timeout)
    stale_time = time.time() - 0.400
    status = safety.evaluate_safety(
        target_q=target_q,
        current_q=current_q,
        cartesian_pos=np.array([0.1, 0.1, 0.5], dtype=np.float32),
        last_packet_time=stale_time,
        dt=0.033
    )

    assert not status.is_safe
    assert status.is_e_stopped
    assert "HEARTBEAT_LOSS" in status.warning_flags
    # Verified velocity output safely ramped down to zero
    assert np.allclose(status.clamped_joint_velocities, 0.0)


def test_safety_monitor_collision_proximity_warning():
    """Verify SafetyMonitor detects obstacle proximity and triggers soft stop."""
    safety = SafetyMonitor(dof=7, min_clearance_meters=0.020)
    
    # Create obstacle at [0.10, 0.10, 0.50] with size [0.08, 0.08, 0.08]
    obstacle = BoundingBox3D(
        center=np.array([0.10, 0.10, 0.50], dtype=np.float32),
        size=np.array([0.08, 0.08, 0.08], dtype=np.float32),
        rotation=np.zeros(3, dtype=np.float32),
        label="coffee cup"
    )

    # Place hand at [0.10, 0.10, 0.54] (distance = 4cm, surface clearance = 4cm - 4cm = 0cm < 2cm)
    hand_pos_near = np.array([0.10, 0.10, 0.54], dtype=np.float32)
    
    status = safety.evaluate_safety(
        target_q=np.zeros(7),
        current_q=np.zeros(7),
        cartesian_pos=hand_pos_near,
        last_packet_time=time.time(),
        obstacles=[obstacle],
        dt=0.033
    )

    assert not status.is_safe
    assert "COLLISION_IMMINENT" in status.warning_flags
    assert status.min_obstacle_clearance_meters < 0.020


def test_stress_noisy_audio_and_fallback():
    """Verify pipeline resilience against corrupt or noisy voice transcriptions."""
    intent_parser = MockLLMIntentParser()

    # Corrupt / empty inputs
    intents = ["", "   ", "???!!!", "random garbled noise", "none", "idle"]
    for raw in intents:
        parsed = intent_parser.parse_intent(raw)
        assert isinstance(parsed, ParsedIntent)
        assert not parsed.is_active or parsed.target_object in ("none", "target_object", "")


def test_stress_tracking_loss_mid_workflow():
    """Verify DiscrepancyEngine and Policy handle sudden hand tracking loss without crashing."""
    engine = DiscrepancyEngine()
    policy = MockResidualPolicy()
    
    # Evaluate with None real_hand (tracking lost)
    disc_state = engine.evaluate(
        real_hand=None,
        foreseen_step=None,
        target_object=None,
        last_action=np.zeros(7, dtype=np.float32)
    )
    assert disc_state.state_vector.shape == (144,)
    assert -1.0 <= disc_state.reward <= 1.0

    action = policy.evaluate(disc_state.state_vector)
    assert action.joint_residuals.shape == (7,)

