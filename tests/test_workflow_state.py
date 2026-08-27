"""Unit and integration tests for Phase 7 Workflow State Machine and Episode Discrepancy Compiler."""

import time
import numpy as np
import pytest

from src.perception.hand_tracker import HandPose, HandSide
from src.perception.intent_parser import MockLLMIntentParser
from src.simulation.trajectory_generator import ForeseenTrajectory, ForeseenWaypoint
from src.policy.discrepancy import DiscrepancyEngine, EpisodeDiscrepancyReport
from src.policy.workflow_state import ExecutionPhase, WorkflowController, WorkflowControlSignal
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.mocks.mock_policy import MockResidualPolicy
from src.transport.ws_server import WSInferenceServer
from src.transport.ws_client import WSStreamingClient


def test_workflow_controller_transitions():
    # step_foresee() advances by elapsed wall-clock time (not call count), so a real
    # rollout duration is used here and driven past completion with a short sleep.
    wf = WorkflowController(foresee_steps=10, foresee_duration_sec=0.05, wait_user_timeout=0.1, execution_max_steps=10, auto_advance=True)
    assert wf.current_phase == ExecutionPhase.IDLE

    # Trigger intent
    wf.trigger_intent("remote_control")
    assert wf.current_phase == ExecutionPhase.FORESEEING
    assert wf.phase_progress == 0.0

    # Step through foresee phase
    time.sleep(0.06)
    wf.step_foresee()

    assert wf.current_phase == ExecutionPhase.WAIT_USER
    time.sleep(0.12)
    wf.step_wait_user()
    assert wf.current_phase == ExecutionPhase.USER_EXECUTING

    # Step through user execution
    dummy_pose = HandPose(
        hand_id=0,
        side=HandSide.RIGHT,
        keypoints_3d=np.zeros((21, 3), dtype=np.float32),
        keypoints_2d=np.zeros((21, 2), dtype=np.float32),
        confidence=0.9,
        timestamp=time.time()
    )

    for _ in range(10):
        wf.record_execution_step(dummy_pose)

    assert wf.current_phase == ExecutionPhase.ADAPTING

    # Reset
    wf.handle_control_command(WorkflowControlSignal.RESET_IDLE.value)
    assert wf.current_phase == ExecutionPhase.IDLE


def test_episode_discrepancy_compiler():
    engine = DiscrepancyEngine()
    policy = MockResidualPolicy()

    # Create synthetic 60-step foreseen trajectory
    waypoints = []
    for t in range(60):
        kpts_3d = np.zeros((21, 3), dtype=np.float32)
        kpts_3d[:, 2] = 0.50 - 0.002 * t
        wp = ForeseenWaypoint(
            timestep=t,
            time_offset=t / 30.0,
            hand_keypoints_3d=kpts_3d,
            hand_keypoints_2d=np.zeros((21, 2), dtype=np.float32),
            wrist_pose=np.array([0.05, 0.10, 0.50, 0.0, 0.0, 0.0], dtype=np.float32),
            object_pose=np.array([0.05, 0.10, 0.50, 0.0, 0.0, 0.0], dtype=np.float32),
            contact_state=np.zeros(5, dtype=np.float32),
            gripper_aperture=0.0
        )
        waypoints.append(wp)

    foreseen_traj = ForeseenTrajectory(
        intent="foresee me picking this remote control",
        target_label="remote_control",
        waypoints=waypoints
    )

    # Create 40-step real hand sequence (simulating faster execution)
    real_poses = []
    for k in range(40):
        kpts_real = np.zeros((21, 3), dtype=np.float32)
        kpts_real[:, 2] = 0.50 - 0.002 * (k * 1.5) + np.random.normal(0, 0.005, size=(21,))
        p = HandPose(
            hand_id=0,
            side=HandSide.RIGHT,
            keypoints_3d=kpts_real,
            keypoints_2d=np.zeros((21, 2), dtype=np.float32),
            confidence=0.95,
            timestamp=time.time()
        )
        real_poses.append(p)

    report = engine.compile_episode_discrepancy(foreseen_traj, real_poses, policy=policy)

    assert isinstance(report, EpisodeDiscrepancyReport)
    assert report.num_steps_sim == 60
    assert report.num_steps_real == 40
    assert report.mean_pose_error >= 0.0
    assert -1.0 <= report.episode_reward <= 1.0
    assert report.policy_loss_delta >= 0.0


@pytest.mark.asyncio
async def test_phase7_websocket_workflow_e2e():
    port = 8799
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
        policy=MockResidualPolicy()
    )

    await server.start()
    client = WSStreamingClient(host="127.0.0.1", port=port)

    try:
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)

        # 1. Send frame with intent to trigger FORESEEING
        resp_1 = await client.send_frame(
            frame,
            frame_id=1,
            intent="foresee me picking this remote control"
        )
        assert resp_1 is not None
        assert resp_1.workflow_phase in ("FORESEEING", "WAIT_USER", "USER_EXECUTING", "IDLE")

        # 2. Advance workflow
        server.workflow.transition_to(ExecutionPhase.USER_EXECUTING)

        resp_2 = await client.send_frame(
            frame,
            frame_id=2,
            intent="foresee me picking this remote control"
        )
        assert resp_2 is not None
        assert resp_2.workflow_phase == "USER_EXECUTING"

        # 3. Complete episode and compile discrepancy
        server.workflow.transition_to(ExecutionPhase.ADAPTING)
        resp_3 = await client.send_frame(
            frame,
            frame_id=3,
            intent="foresee me picking this remote control"
        )
        assert resp_3 is not None
        assert resp_3.episode_report is not None
        assert "mean_pose_error" in resp_3.episode_report
    finally:
        await client.close()
        await server.stop()
