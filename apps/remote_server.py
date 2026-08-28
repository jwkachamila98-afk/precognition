"""Remote backend inference server (apps/remote_server.py).

Runs an async WebSocket server on port 8765, executing the full Phase 8 visuomotor pipeline:
- MockHandTracker / MediaPipeHandTracker
- MockDepthEstimator
- StructuredLLMIntentParser (real local LLM via Ollama, rule-based fallback per-request)
- WorkflowController (Staged Foresee-then-Execute state machine)
- MockSceneParser (Grounded 3D bounding boxes)
- MockAffordanceExtractor (Contact probability maps)
- MockTrajectoryDiffusion (60-step foreseen reference trajectory rollout)
- DiscrepancyEngine (112D state vector & episode discrepancy compilation)
- NeuralResidualPolicy (real online-learning MLP, Reward-Weighted Regression -> MockResidualPolicy fallback)
- MockRobotHardware (7-DOF robotic manipulator & actuator dynamic constraints)
- PolicyCheckpointManager (Persistent profile & weight storage)
- CoAdaptationBenchmark (Multi-trial analytics & convergence curves)
- MockPhysicsEngine (analytical contact states)
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import urllib.request
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_parser import AppConfig
# Must run before torch/ultralytics are imported: they download models over
# HTTPS and cannot be handed a custom SSL context (see src/utils/certs.py).
from src.utils.certs import ensure_ca_bundle

ensure_ca_bundle()

from src.perception.intent_parser import StructuredLLMIntentParser
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

    # --- Intent parsing: real local LLM (Ollama) -> rule-based parser per-request fallback ---
    ollama_url = "http://localhost:11434/api/generate"
    ollama_model = "llama3.2:1b"
    try:
        probe = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2.0)
        probe.read()
        logger.info(f"RemoteServer: Ollama reachable. Using StructuredLLMIntentParser ({ollama_model}).")
    except Exception as e:
        logger.warning(
            f"RemoteServer: Ollama not reachable at startup ({e}). StructuredLLMIntentParser "
            "will fall back to the rule-based parser per-request until it comes up."
        )
    intent_parser = StructuredLLMIntentParser(
        endpoint_url=ollama_url, model_name=ollama_model, timeout=8.0
    )

    # --- Open-vocabulary fallback: Gemini vision grounding for descriptions that
    # don't resolve to a COCO-80 class (only called on intent change, see LiveSceneParser) ---
    vision_grounder = None
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            from src.perception.gemini_vision_grounder import GeminiVisionGrounder
            vision_grounder = GeminiVisionGrounder(api_key=gemini_api_key)
            logger.info(f"RemoteServer: GEMINI_API_KEY found. Open-vocabulary vision grounding enabled ({vision_grounder.model}).")
        except Exception as e:
            logger.warning(f"GeminiVisionGrounder initialization failed: {e}")
    else:
        logger.info("RemoteServer: GEMINI_API_KEY not set. Open-vocabulary vision grounding disabled (COCO-80 only).")

    # --- Object detection: GPU YOLO -> CPU MediaPipe EfficientDet -> mock canned bbox ---
    scene_parser = None
    try:
        from src.perception.yolo_object_detector import YoloObjectDetector
        from src.perception.live_scene_parser import LiveSceneParser
        # yolov8s on a GPU costs nothing; on a CPU-only host it is ~0.74 s/frame,
        # which paces the whole workflow. YOLO_MODEL lets a slow host drop to
        # yolov8n without changing what a GPU deployment runs.
        yolo_model = os.environ.get("YOLO_MODEL", "yolov8s.pt")
        object_detector = YoloObjectDetector(model_name=yolo_model, conf_threshold=0.30)
        scene_parser = LiveSceneParser(
            object_detector=object_detector,
            num_points=app_config.perception.scene_parser.num_points,
            vision_grounder=vision_grounder
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
                num_points=app_config.perception.scene_parser.num_points,
                vision_grounder=vision_grounder
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

    # --- Residual policy: real online-learning neural network (GPU) -> fixed
    # linear feedback fallback (CPU-only environments without torch) ---
    try:
        from src.policy.neural_policy import NeuralResidualPolicy
        policy = NeuralResidualPolicy()
        logger.info(
            f"RemoteServer: Using real NeuralResidualPolicy (device={policy.device}), "
            "trained online via Reward-Weighted Regression."
        )
    except ImportError as e:
        logger.warning(f"NeuralResidualPolicy unavailable ({e}); falling back to MockResidualPolicy.")
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
