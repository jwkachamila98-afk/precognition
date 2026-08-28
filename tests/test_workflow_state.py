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


def test_autonomous_demo_can_be_replayed_without_re_arming_the_intent():
    """The demo must be repeatable while the target is still in frame.

    Regression from a live run: the demo ends by returning to IDLE, and IDLE
    cleared the target label, so the guard in handle_control_command refused
    every press after the first - silently, with the object still detected and
    the intent unchanged.
    """
    wf = WorkflowController(auto_advance=False)
    wf.trigger_intent("coffee cup")
    assert wf._target_label == "coffee cup"

    wf.handle_control_command("START_AUTONOMOUS_DEMO")
    assert wf.current_phase == ExecutionPhase.AUTONOMOUS_DEMO

    wf.transition_to(ExecutionPhase.IDLE)          # how the demo ends
    assert wf._target_label == "coffee cup", "the showcase must not clear the intent"

    wf.handle_control_command("START_AUTONOMOUS_DEMO")
    assert wf.current_phase == ExecutionPhase.AUTONOMOUS_DEMO, "second demo was refused"


def test_withdrawing_the_intent_still_clears_the_target():
    """The exception above is scoped to the demo: an explicit intent withdrawal
    (or any other route to IDLE) must still clear the target."""
    wf = WorkflowController(auto_advance=False)
    wf.trigger_intent("coffee cup")

    wf.transition_to(ExecutionPhase.USER_EXECUTING)
    wf.transition_to(ExecutionPhase.IDLE)
    assert wf._target_label == "none"

    wf.trigger_intent("coffee cup")
    wf.trigger_intent("none")
    assert wf._target_label == "none"
    assert wf.current_phase == ExecutionPhase.IDLE


def test_autonomous_demo_request_is_deferred_not_dropped():
    """A demo asked for mid-episode must run when a phase will accept it.

    Regression from a live run. The request travels a frame behind the keypress
    and the phases auto-advance on their own timers, so on a slow host the
    workflow routinely moves into a refusing phase in that gap. Pressing the key
    then did nothing whatsoever - indistinguishable from a broken feature. It
    was hit four times in twenty seconds before the user gave up.
    """
    wf = WorkflowController(auto_advance=False)
    wf.trigger_intent("coffee cup")
    wf.transition_to(ExecutionPhase.USER_EXECUTING)

    wf.handle_control_command("START_AUTONOMOUS_DEMO")
    assert wf.current_phase == ExecutionPhase.USER_EXECUTING, "must not barge in mid-attempt"
    assert wf.poll_pending_demo() is False, "still executing; nothing to start yet"

    wf.transition_to(ExecutionPhase.ADAPTING)
    assert wf.poll_pending_demo() is False, "still adapting; nothing to start yet"

    wf.transition_to(ExecutionPhase.WAIT_USER)
    assert wf.poll_pending_demo() is True
    assert wf.current_phase == ExecutionPhase.AUTONOMOUS_DEMO


def test_deferred_demo_request_expires():
    """A swallowed request must not surprise the user minutes later."""
    wf = WorkflowController(auto_advance=False)
    wf.pending_demo_ttl_sec = 0.05
    wf.trigger_intent("coffee cup")
    wf.transition_to(ExecutionPhase.USER_EXECUTING)
    wf.handle_control_command("START_AUTONOMOUS_DEMO")

    time.sleep(0.08)
    wf.transition_to(ExecutionPhase.WAIT_USER)
    assert wf.poll_pending_demo() is False
    assert wf.current_phase == ExecutionPhase.WAIT_USER


def test_demo_available_immediately_when_the_phase_allows():
    """The deferral path must not slow down the normal case."""
    wf = WorkflowController(auto_advance=False)
    wf.trigger_intent("coffee cup")
    wf.transition_to(ExecutionPhase.WAIT_USER)
    wf.handle_control_command("START_AUTONOMOUS_DEMO")
    assert wf.current_phase == ExecutionPhase.AUTONOMOUS_DEMO
    assert wf._pending_demo_at is None


def test_deferred_demo_survives_an_attempt_longer_than_the_ttl():
    """A real attempt runs far longer than the expiry window.

    Regression from a live run: the TTL was wall-clock from the keypress, so a
    44-second execution phase discarded the request every time and the user saw
    nothing - the same silent failure the deferral was added to fix. The clock
    must only run while a phase would actually accept the demo.
    """
    wf = WorkflowController(auto_advance=False)
    wf.pending_demo_ttl_sec = 0.05
    wf.trigger_intent("coffee cup")
    wf.transition_to(ExecutionPhase.USER_EXECUTING)
    wf.handle_control_command("START_AUTONOMOUS_DEMO")

    # Far longer than the TTL, but all of it inside a phase that refuses.
    for _ in range(4):
        time.sleep(0.03)
        assert wf.poll_pending_demo() is False
    assert wf._pending_demo_at is not None, "request must survive a long attempt"

    wf.transition_to(ExecutionPhase.WAIT_USER)
    assert wf.poll_pending_demo() is True
    assert wf.current_phase == ExecutionPhase.AUTONOMOUS_DEMO


def test_deferred_demo_still_expires_while_startable():
    """The expiry must still fire when the demo could have run but did not."""
    wf = WorkflowController(auto_advance=False)
    wf.pending_demo_ttl_sec = 0.05
    wf.trigger_intent("coffee cup")
    wf.transition_to(ExecutionPhase.USER_EXECUTING)
    wf.handle_control_command("START_AUTONOMOUS_DEMO")
    wf.poll_pending_demo()

    wf._phase = ExecutionPhase.WAIT_USER      # startable, but never polled
    time.sleep(0.08)
    assert wf.poll_pending_demo() is False
    assert wf.current_phase == ExecutionPhase.WAIT_USER


def test_restart_phase_timer_extends_the_current_phase():
    """A phase that had to wait for something before it could begin must still
    get its full duration for the part the user actually sees."""
    wf = WorkflowController(auto_advance=True, autonomous_demo_duration_sec=0.20)
    wf.trigger_intent("coffee cup")
    wf.transition_to(ExecutionPhase.AUTONOMOUS_DEMO)

    time.sleep(0.15)                       # spent waiting for a target
    wf.restart_phase_timer()
    assert wf.step_autonomous_demo() is False, "clock should have restarted"
    assert wf.current_phase == ExecutionPhase.AUTONOMOUS_DEMO
    assert wf.phase_progress < 0.5

    time.sleep(0.25)
    assert wf.step_autonomous_demo() is True
    assert wf.current_phase == ExecutionPhase.IDLE


def test_second_demo_request_abandons_the_attempt():
    """Asking twice means "now", not "queue it again".

    USER_EXECUTING ends after 60 DETECTED hand poses rather than on a timer, so
    it runs about a minute with the hand in view and stalls indefinitely without
    it. Deferring politely to that is not what a second keypress is asking for -
    observed live, the user pressed twice on each of two consecutive attempts.
    """
    wf = WorkflowController(auto_advance=False)
    wf.trigger_intent("coffee cup")
    wf.transition_to(ExecutionPhase.USER_EXECUTING)

    wf.handle_control_command("START_AUTONOMOUS_DEMO")
    assert wf.current_phase == ExecutionPhase.USER_EXECUTING
    assert wf._pending_demo_at is not None

    wf.handle_control_command("START_AUTONOMOUS_DEMO")
    assert wf.current_phase == ExecutionPhase.AUTONOMOUS_DEMO
    assert wf._pending_demo_at is None, "the pending request was consumed, not left armed"


def test_single_request_still_waits_politely():
    """One press must not interrupt a real attempt."""
    wf = WorkflowController(auto_advance=False)
    wf.trigger_intent("coffee cup")
    wf.transition_to(ExecutionPhase.USER_EXECUTING)
    wf.handle_control_command("START_AUTONOMOUS_DEMO")
    assert wf.current_phase == ExecutionPhase.USER_EXECUTING
