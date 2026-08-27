"""Remote backend inference server (apps/remote_server.py).

Runs an async WebSocket server on port 8765, executing the full Phase 8 visuomotor pipeline:
- MockHandTracker / MediaPipeHandTracker
- MockDepthEstimator
- MockLLMIntentParser (Structured semantic schema grounding)
- WorkflowController (Staged Foresee-then-Execute state machine)
- MockSceneParser (Grounded 3D bounding boxes)
- MockAffordanceExtractor (Contact probability maps)
- MockTrajectoryDiffusion (60-step foreseen reference trajectory rollout)
- DiscrepancyEngine (112D state vector & episode discrepancy compilation)
- MockResidualPolicy (online PPO learning loop & residual joint corrections)
- MockRobotHardware (7-DOF robotic manipulator & actuator dynamic constraints)
- PolicyCheckpointManager (Persistent profile & weight storage)
- CoAdaptationBenchmark (Multi-trial analytics & convergence curves)
- MockPhysicsEngine (analytical contact states)
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_parser import AppConfig
from src.perception.intent_parser import MockLLMIntentParser
from src.mocks.mock_hand_tracker import MockHandTracker
from src.mocks.mock_depth_estimator import MockDepthEstimator
from src.mocks.mock_scene_parser import MockSceneParser
from src.mocks.mock_affordance_extractor import MockAffordanceExtractor
from src.mocks.mock_trajectory_diffusion import MockTrajectoryDiffusion
from src.mocks.mock_physics_engine import MockPhysicsEngine
from src.mocks.mock_policy import MockResidualPolicy
from src.policy.discrepancy import DiscrepancyEngine
from src.policy.workflow_state import WorkflowController
from src.policy.checkpointing import PolicyCheckpointManager
from src.analytics.benchmark import CoAdaptationBenchmark
from src.hardware.robot_interface import MockRobotHardware
from src.transport.ws_server import WSInferenceServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RemoteServer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visuomotor Hand Policy Remote Inference Server (Phase 8)")
    parser.add_argument("--config", type=str, default="config/system_config.yaml", help="Path to config YAML")
    parser.add_argument("--host", type=str, default=None, help="Host address to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default 8765)")
    parser.add_argument("--tracker", type=str, default="mediapipe", choices=["mediapipe", "mock"], help="Hand tracker backend")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    app_config = AppConfig.from_yaml(config_path)
    host = args.host or "0.0.0.0"
    port = args.port or app_config.network.server_port

    logger.info("Initializing Phase 8 perception, LLM intent reasoning, robot hardware, and co-adaptation analytics...")

    # --- Hand tracking: Tasks API (works on any mediapipe version) -> legacy solutions API -> mock ---
    hand_tracker = None
    if args.tracker == "mediapipe":
        try:
            from src.perception.mediapipe_tasks_hand_tracker import MediaPipeTasksHandTracker
            hand_tracker = MediaPipeTasksHandTracker()
            logger.info("RemoteServer: Using live MediaPipeTasksHandTracker (real hand tracking).")
        except Exception as e:
            logger.warning(f"MediaPipe Tasks hand tracker initialization failed: {e}")
            try:
                from src.perception.mediapipe_tracker import MediaPipeHandTracker, MEDIAPIPE_AVAILABLE
                if MEDIAPIPE_AVAILABLE:
                    hand_tracker = MediaPipeHandTracker()
                    logger.info("RemoteServer: Using live MediaPipeHandTracker (legacy solutions API).")
            except Exception as e2:
                logger.warning(f"Legacy MediaPipe hand tracker initialization also failed: {e2}")
    if hand_tracker is None:
        hand_tracker = MockHandTracker()
        logger.warning("RemoteServer: Falling back to MockHandTracker (no real hand tracking available).")

    # --- Depth: real MiDaS monocular depth on GPU -> mock synthetic depth ---
    try:
        from src.perception.midas_depth_estimator import MiDaSDepthEstimator
        depth_estimator = MiDaSDepthEstimator(
            min_depth=app_config.perception.depth_estimator.min_depth_meters,
            max_depth=app_config.perception.depth_estimator.max_depth_meters,
        )
        logger.info("RemoteServer: Using live MiDaSDepthEstimator (real monocular depth).")
    except Exception as e:
        logger.warning(f"MiDaS depth estimator initialization fallback: {e}")
        depth_estimator = MockDepthEstimator(
            min_depth=app_config.perception.depth_estimator.min_depth_meters,
            max_depth=app_config.perception.depth_estimator.max_depth_meters,
            target_shape=(
                app_config.perception.depth_estimator.output_height,
                app_config.perception.depth_estimator.output_width
            )
        )

    intent_parser = MockLLMIntentParser()

    # --- Object detection: GPU YOLO -> CPU MediaPipe EfficientDet -> mock canned bbox ---
    scene_parser = None
    try:
        from src.perception.yolo_object_detector import YoloObjectDetector
        from src.perception.live_scene_parser import LiveSceneParser
        object_detector = YoloObjectDetector(model_name="yolov8s.pt", conf_threshold=0.30)
        scene_parser = LiveSceneParser(
            object_detector=object_detector,
            num_points=app_config.perception.scene_parser.num_points
        )
        logger.info("RemoteServer: Using live YoloObjectDetector-backed LiveSceneParser (real GPU object recognition).")
    except Exception as e:
        logger.warning(f"YOLO object detector initialization failed: {e}")
        try:
            from src.perception.object_detector import MediaPipeObjectDetector
            from src.perception.live_scene_parser import LiveSceneParser
            object_detector = MediaPipeObjectDetector(score_threshold=0.30)
            scene_parser = LiveSceneParser(
                object_detector=object_detector,
                num_points=app_config.perception.scene_parser.num_points
            )
            logger.info("RemoteServer: Using live MediaPipeObjectDetector-backed LiveSceneParser (real CPU object recognition).")
        except Exception as e2:
            logger.warning(f"MediaPipe object detector initialization also failed: {e2}")
    if scene_parser is None:
        scene_parser = MockSceneParser(
            num_points=app_config.perception.scene_parser.num_points
        )
        logger.warning("RemoteServer: Falling back to MockSceneParser (no real object recognition available).")
    affordance_extractor = MockAffordanceExtractor()
    trajectory_diffusion = MockTrajectoryDiffusion()
    discrepancy_engine = DiscrepancyEngine()
    physics_engine = MockPhysicsEngine()
    policy = MockResidualPolicy()
    workflow = WorkflowController(foresee_steps=60, wait_user_timeout=2.0, auto_advance=True)
    robot = MockRobotHardware(dof=7)
    checkpoint_manager = PolicyCheckpointManager()
    benchmark = CoAdaptationBenchmark()

    # Pre-register procedural objects into physics engine
    physics_engine.instantiate_object_mesh("remote_control", position=np.array([0.08, 0.12, 0.58], dtype=np.float32))

    server = WSInferenceServer(
        host=host,
        port=port,
        hand_tracker=hand_tracker,
        depth_estimator=depth_estimator,
        scene_parser=scene_parser,
        intent_parser=intent_parser,
        affordance_extractor=affordance_extractor,
        trajectory_diffusion=trajectory_diffusion,
        discrepancy_engine=discrepancy_engine,
        physics_engine=physics_engine,
        policy=policy,
        workflow=workflow,
        robot=robot,
        checkpoint_manager=checkpoint_manager,
        benchmark=benchmark
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_server() -> None:
        await server.start()
        stop_event = asyncio.Event()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass

        logger.info(f"Phase 8 Server active on ws://{host}:{port}. Ready for local_client.py connections.")
        await stop_event.wait()
        logger.info("Shutdown initiated...")
        await server.stop()

    try:
        loop.run_until_complete(run_server())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        loop.close()
        logger.info("Remote server stopped.")


if __name__ == "__main__":
    main()
