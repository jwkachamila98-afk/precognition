"""Async WebSocket server (ws_server.py) for remote inference and Phase 9 Safety Guardrails."""

import asyncio
import logging
import time
from typing import Any, Optional
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from src.perception.hand_tracker import HandTrackerABC
from src.perception.depth_estimator import DepthEstimatorABC
from src.perception.scene_parser import SceneParserABC
from src.perception.intent_parser import IntentParserABC, MockLLMIntentParser, ParsedIntent
from src.simulation.simulator import SimAction, SimulatorABC
from src.simulation.trajectory_generator import TrajectoryGeneratorABC
from src.policy.policy import PolicyABC
from src.policy.discrepancy import DiscrepancyEngine, DiscrepancyEngineABC, EpisodeDiscrepancyReport
from src.policy.workflow_state import ExecutionPhase, WorkflowController
from src.policy.checkpointing import PolicyCheckpointManager
from src.analytics.benchmark import CoAdaptationBenchmark
from src.hardware.robot_interface import MockRobotHardware, RobotHardwareABC, RobotState
from src.safety.safety_monitor import SafetyMonitor, SafetyStatus
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.mocks.mock_policy import MockResidualPolicy
from src.transport.protocol import (
    FrameMessage,
    InferenceResponse,
    encode_image_to_base64,
)

logger = logging.getLogger(__name__)


class WSInferenceServer:
    """
    Async WebSocket backend server for Phase 9.
    Integrates the Full Visuomotor Pipeline with Real-Time Safety Guardrails,
    Kinematic Joint & Workspace Limits, Collision Interlocks, and Heartbeat Drop Monitoring.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        hand_tracker: Optional[HandTrackerABC] = None,
        depth_estimator: Optional[DepthEstimatorABC] = None,
        scene_parser: Optional[SceneParserABC] = None,
        intent_parser: Optional[IntentParserABC] = None,
        affordance_extractor: Optional[MockAffordanceExtractor] = None,
        trajectory_diffusion: Optional[TrajectoryGeneratorABC] = None,
        discrepancy_engine: Optional[DiscrepancyEngineABC] = None,
        physics_engine: Optional[SimulatorABC] = None,
        policy: Optional[PolicyABC] = None,
        workflow: Optional[WorkflowController] = None,
        robot: Optional[RobotHardwareABC] = None,
        checkpoint_manager: Optional[PolicyCheckpointManager] = None,
        benchmark: Optional[CoAdaptationBenchmark] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.hand_tracker = hand_tracker or MockHandTracker()
        self.depth_estimator = depth_estimator or MockDepthEstimator()
        self.scene_parser = scene_parser or MockSceneParser()
        self.intent_parser = intent_parser or MockLLMIntentParser()
        self.affordance_extractor = affordance_extractor or MockAffordanceExtractor()
        self.trajectory_diffusion = trajectory_diffusion or MockTrajectoryDiffusion()
        self.discrepancy_engine = discrepancy_engine or DiscrepancyEngine()
        self.physics_engine = physics_engine or MockPhysicsEngine()
        self.policy = policy or MockResidualPolicy()
        self.workflow = workflow or WorkflowController(foresee_steps=60, wait_user_timeout=2.0)
        self.robot = robot or MockRobotHardware()
        self.checkpoint_manager = checkpoint_manager or PolicyCheckpointManager()
        self.benchmark = benchmark or CoAdaptationBenchmark()
        self.safety_monitor = safety_monitor or SafetyMonitor()

        self._server = None
        self._is_running = False
        self._last_action = np.zeros(7, dtype=np.float32)
        self._last_intent = ""
        self._cached_parsed_intent = self.intent_parser.parse_intent("")
        self._cached_foreseen_traj = None
        self._last_client_time = time.time()

    async def handle_client(self, websocket: Any) -> None:
        client_address = getattr(websocket, "remote_address", "client")
        logger.info(f"Client connected from {client_address}")

        try:
            async for raw_message in websocket:
                t0 = time.perf_counter()
                
                # Parse incoming frame message
                frame_msg = FrameMessage.from_json(raw_message)
                image = frame_msg.decode_image()

                if image is None:
                    logger.warning("Failed to decode frame image. Skipping.")
                    continue

                # Handle client control commands
                if frame_msg.control_command:
                    cmd = frame_msg.control_command.upper()
                    if cmd == "SAVE_CHECKPOINT":
                        self.checkpoint_manager.save_checkpoint(self.policy)
                    elif cmd == "LOAD_CHECKPOINT":
                        self.checkpoint_manager.load_checkpoint(self.policy)
                    elif cmd == "RESET_BASELINE":
                        self.checkpoint_manager.reset_to_baseline(self.policy)
                    else:
                        self.workflow.handle_control_command(cmd)

                # 1. Perception: Hand tracking & Depth estimation
                hand_poses = self.hand_tracker.estimate(image)
                depth_map = self.depth_estimator.estimate_depth(image)

                # 2. Perception: Structured Intent Parsing & Change Detection.
                # Only re-parse (and re-trigger the workflow) when the intent text actually
                # changes - critical when intent_parser is a real LLM backend, since a raw
                # intent call every single frame would be both wasteful and far too slow.
                if frame_msg.intent != self._last_intent:
                    self._last_intent = frame_msg.intent
                    parsed_intent = self.intent_parser.parse_intent(frame_msg.intent)
                    self._cached_parsed_intent = parsed_intent
                    target_label = parsed_intent.target_object if parsed_intent.is_active else "none"
                    self.workflow.trigger_intent(target_label)
                else:
                    parsed_intent = self._cached_parsed_intent

                # 4. Perception: Grounded 3D Scene Parsing
                parsed_scene = self.scene_parser.parse_scene(
                    image=image,
                    depth=depth_map,
                    intent=frame_msg.intent
                )

                # 5. Affordance & Foreseen Trajectory Rollout
                affordance_dict = None
                foreseen_dict = None
                current_foreseen_step = None
                target_box = None
                episode_report_dict = None

                if parsed_scene.bounding_boxes:
                    target_box = parsed_scene.bounding_boxes[0]
                    affordance_map = self.affordance_extractor.extract_affordance(
                        bounding_box=target_box,
                        intent=frame_msg.intent
                    )
                    affordance_dict = affordance_map.to_dict()

                    start_pose = hand_poses[0] if hand_poses else None
                    if self._cached_foreseen_traj is None or self.workflow.current_phase == ExecutionPhase.FORESEEING:
                        self._cached_foreseen_traj = self.trajectory_diffusion.generate_foreseen_rollout(
                            start_hand_pose=start_pose,
                            target_object=target_box,
                            affordance_map=affordance_map,
                            intent=frame_msg.intent,
                            num_steps=60
                        )
                        self.workflow.stored_foreseen_trajectory = self._cached_foreseen_traj

                    foreseen_dict = self._cached_foreseen_traj.to_dict()
                    step_idx = self.workflow.step_index % len(self._cached_foreseen_traj.waypoints)
                    current_foreseen_step = self._cached_foreseen_traj.waypoints[step_idx]

                # 6. Workflow State Machine Lifecycle Stepping
                current_phase = self.workflow.current_phase

                if current_phase == ExecutionPhase.FORESEEING:
                    self.workflow.step_foresee()
                elif current_phase == ExecutionPhase.WAIT_USER:
                    self.workflow.step_wait_user()
                elif current_phase == ExecutionPhase.USER_EXECUTING:
                    real_h = hand_poses[0] if hand_poses else None
                    self.workflow.record_execution_step(real_h)
                elif current_phase == ExecutionPhase.ADAPTING:
                    rep = self.discrepancy_engine.compile_episode_discrepancy(
                        foreseen_traj=self.workflow.stored_foreseen_trajectory,
                        recorded_poses=self.workflow.recorded_physical_poses,
                        policy=self.policy
                    )
                    episode_report_dict = rep.to_dict()
                    self.workflow.last_adaptation_report = episode_report_dict
                    self.benchmark.record_trial(rep, intent=frame_msg.intent)
                    logger.info(f"Episode Discrepancy Compiled: Reward={rep.episode_reward:+.3f} | MSE={rep.mean_pose_error:.4f}m")
                    self.workflow.transition_to(ExecutionPhase.IDLE)
                    self._cached_foreseen_traj = None

                # 7. Discrepancy Engine & Policy Evaluation
                real_h = hand_poses[0] if hand_poses else None
                discrepancy_state = self.discrepancy_engine.evaluate(
                    real_hand=real_h,
                    foreseen_step=current_foreseen_step,
                    target_object=target_box,
                    last_action=self._last_action
                )

                action = self.policy.evaluate(discrepancy_state.state_vector)
                self._last_action = action.joint_residuals.copy()

                if hasattr(self.policy, "record_transition") and getattr(self.policy, "adaptation_active", True):
                    self.policy.record_transition(
                        state=discrepancy_state.state_vector,
                        action=action.joint_residuals,
                        reward=discrepancy_state.reward
                    )

                # 8. Real-Time Safety Guardrail Evaluation
                cur_robot_state = self.robot.read_joint_states()
                cart_pos = real_h.keypoints_3d[0] if real_h else np.array([0.05, 0.10, 0.50], dtype=np.float32)
                
                target_joints = np.zeros(7, dtype=np.float32)
                if current_foreseen_step is not None:
                    target_joints[:min(6, len(target_joints))] = current_foreseen_step.wrist_pose[:min(6, len(target_joints))]
                    target_joints[:len(action.joint_residuals)] += action.joint_residuals

                safety = self.safety_monitor.evaluate_safety(
                    target_q=target_joints,
                    current_q=cur_robot_state.joint_positions,
                    cartesian_pos=cart_pos,
                    last_packet_time=self._last_client_time,
                    obstacles=parsed_scene.bounding_boxes,
                    dt=0.033
                )
                self._last_client_time = time.time()

                # 9. Robot Hardware Actuation Step
                self.robot.send_joint_commands(safety.clamped_joint_positions, gripper_command=action.gripper_action)
                robot_telemetry = self.robot.read_joint_states()

                # 10. Physics Engine Step
                sim_action = SimAction(
                    target_joint_positions=None,
                    gripper_command=action.gripper_action
                )
                self.physics_engine.step(sim_action)

                # 11. Render depth heatmap
                depth_heatmap = depth_map.to_colored_heatmap()
                depth_b64 = encode_image_to_base64(depth_heatmap, quality=60)

                proc_time_ms = (time.perf_counter() - t0) * 1000.0

                step_cnt = getattr(self.policy, "step_count", 0)
                p_loss = getattr(self.policy, "loss_history", [0.0])[-1]
                adapt_status = "ACTIVE" if getattr(self.policy, "adaptation_active", True) else "PAUSED"

                # Formulate structured JSON response with Phase 9 safety & telemetry metrics
                response = InferenceResponse(
                    frame_id=frame_msg.frame_id,
                    client_timestamp=frame_msg.client_timestamp,
                    server_timestamp=time.time(),
                    hand_poses=[hp.to_dict() for hp in hand_poses],
                    depth_heatmap_base64=depth_b64,
                    parsed_scene=parsed_scene.to_dict(),
                    parsed_intent=parsed_intent.to_dict(),
                    affordance_map=affordance_dict,
                    foreseen_trajectory=foreseen_dict,
                    policy_residuals=action.joint_residuals.tolist(),
                    reward_score=discrepancy_state.reward,
                    discrepancy_norm=discrepancy_state.discrepancy_norm,
                    workflow_phase=self.workflow.current_phase.value,
                    phase_progress=self.workflow.phase_progress,
                    episode_report=episode_report_dict or self.workflow.last_adaptation_report,
                    robot_state=robot_telemetry.to_dict(),
                    safety_status=safety.to_dict(),
                    benchmark_summary=self.benchmark.get_summary(),
                    adaptation_status=adapt_status,
                    buffer_step_count=step_cnt,
                    policy_loss=p_loss,
                    gripper_action=action.gripper_action,
                    server_processing_ms=proc_time_ms
                )

                await websocket.send(response.to_json())

        except ConnectionClosed:
            logger.info(f"Client {client_address} disconnected.")
        except Exception as e:
            logger.error(f"Error processing client stream {client_address}: {e}", exc_info=True)

    async def start(self) -> None:
        """Start async WebSocket server listener."""
        try:
            from websockets.asyncio.server import serve as ws_serve
        except ImportError:
            from websockets import serve as ws_serve

        self.robot.connect()
        self._server = await ws_serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024
        )
        self._is_running = True
        logger.info(f"Phase 9 Inference Server active and listening on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop server gracefully."""
        if self._server:
            self._server.close()
            if hasattr(self._server, "wait_closed"):
                await self._server.wait_closed()
        self.robot.disconnect()
        self._is_running = False
        logger.info("Inference Server stopped.")


# Backwards compatibility alias
WebSocketInferenceServer = WSInferenceServer
