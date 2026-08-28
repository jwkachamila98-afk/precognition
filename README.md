# Visuomotor Hand Policy & Residual Adaptation Architecture

A modular, hardware-decoupled **Visuomotor Hand Policy & Residual Adaptation Architecture** designed for real-time robotic manipulation, 3D hand mesh estimation (MANO), metric depth estimation, continuous speech-to-text audio ingestion (Whisper / Silero VAD), structured LLM intent reasoning, a **Staged 'Foresee-then-Execute' Workflow State Machine**, **Robot Hardware Actuation Bridges (ROS 2 / Mock)**, **Persistent Policy Checkpointing**, **Multi-Trial Co-Adaptation Analytics**, **Real-Time Safety Guardrails & Collision Interlocks**, intent-conditioned 3D spatial scene parsing, physics simulation, online episode discrepancy learning, and cloud GPU deployment.

Runs in real-time ($30+\text{ FPS}$) on resource-constrained **Intel Mac CPU** hardware while remaining 100% plug-and-play compatible with cloud GPU backends (CUDA / TensorRT / MuJoCo MJX on RunPod, Lambda Labs, AWS EC2).

```
                                      COMPLETE SYSTEM ARCHITECTURE
                                  
  +---------------------------------------------------------------------------------------------------+
  |                                LOCAL CLIENT (Intel Mac - CPU Only)                                |
  |                                                                                                   |
  |  +---------------------+        +-----------------------------------+      +------------------+   |
  |  |   Camera Ingestion  |  --->  |         OpenCV Visualizer         | <--- |    WS Client     |   |
  |  |  (Webcam/Synthetic)|        |  2D/3D Skeleton HUD & Color Inset |      |  (ws_client.py)  |   |
  |  +---------------------+        |  3D Bounding Box & Affordance Hot |      +------------------+   |
  |                                 |  'Foreseen' Ghost Hand (tau_ref)  |               ^             |
  |  +---------------------+        |  [1/3] Foresee -> [2/3] Execute   |               |             |
  |  | Audio STT (Whisper/ |  --->  |  [3/3] Discrepancy Adapt Banner   |               |             |
  |  |  MockTranscriber)   |        |  Co-Adaptation Benchmark Panel    |               |             |
  |  +---------------------+        |  Safety Guardrail & Limits Banner |               |             |
  |                                 |  Voice Telemetry & Intent Banner  |               |             |
  |  +---------------------+        |  LatencyProfiler (P99 breakdown)  |               |             |
  |  | Checkpoint Manager  |  --->  |  Session Dataset Recorder (MP4)   |               |             |
  |  | (config/profiles/)  |        +-----------------------------------+               |             |
  |  +---------------------+                          |                                 |             |
  +---------------------------------------------------| + Speech/Intent + Workflow Phase| Telemetry   |
                                                      | Frame (JPEG base64)             | (112D state,|
                                                      v                                 | MANO, Diff, |
  +-------------------------------------------------------------------------------------| Robot, Safe,|
  |                     CLOUD GPU INFERENCE BACKEND (Docker / CUDA 12.2 / TensorRT)     | Bench)      |
  |                                                                                     |             |
  |   +---------------------------------------------------------------------------------+---------+   |
  |   | WebSocket Inference Server (ws_server.py on ws://0.0.0.0:8765)                            |   |
  |   +-------------------------------------------------------------------------------------------+   |
  |            |                       |                      |                        |              |
  |            v                       v                      v                        v              |
  |   +-------------------+  +-------------------+  +--------------------+   +--------------------+   |
  |   | Hand Tracker ABC  |  | Depth Estimator   |  | Structured LLM     |   | WorkflowController |   |
  |   | (MANO 21-Joints / |  | (Metric Depth /   |  | Intent Parser      |   | (Foresee -> Exec ->|   |
  |   |  MediaPipe / Mock)|  |  DepthAnything V2)|  | (Schema Grounding) |   |  Discrepancy Adapt)|   |
  |   +-------------------+  +-------------------+  +--------------------+   +--------------------+   |
  |            |                       |                      |                        |              |
  |            +-----------------------+----------------------+                        v              |
  |                                    |                                     +--------------------+   |
  |                                    v                                     | Discrepancy Engine |   |
  |                         +--------------------+                           | (112D s_t & D_traj |   |
  |                         | MockAffordance &   |                           |  Episode Compiler) |   |
  |                         | TrajectoryDiffusion|                           +--------------------+   |
  |                         | (60-step tau_ref)  |                                     |              |
  |                         +--------------------+                                     v              |
  |                                    |                                     +--------------------+   |
  |                                    |                                     | NeuralPolicy (RWR) |   |
  |                                    |                                     | RWR-trained online |   |
  |                                    |                                     |  & Delta theta_t)  |   |
  |                                    |                                     +--------------------+   |
  |                                    |                                               |              |
  |                                    +---------------------------------------------> v              |
  |                                                                          +--------------------+   |
  |                                                                          | SafetyMonitor      |   |
  |                                                                          | (Limits, Heartbeat,|   |
  |                                                                          |  Collisions)       |   |
  |                                                                          +--------------------+   |
  |                                                                                    |              |
  |                                                                                    v              |
  |                                                                          +--------------------+   |
  |                                                                          | RobotHardwareABC   |   |
  |                                                                          | (7-DOF Arm + Hand  |   |
  |                                                                          |  ROS 2 Bridge)     |   |
  |                                                                          +--------------------+   |
  +---------------------------------------------------------------------------------------------------+
```

---

## Directory Structure

```
Precognition/
├── README.md                          # Full system documentation & usage guide
├── requirements.txt                   # Intel Mac CPU-friendly dependencies
├── .gitignore                         # Git ignore rules
├── apps/
│   ├── run_demo.py                    # Unified Demo Launcher & Automated Startup Entrypoint
│   ├── local_client.py                # Real-time OpenCV client + HUD, robot, analytics & checkpoints
│   └── remote_server.py               # Backend WebSocket inference server
├── config/
│   ├── system_config.yaml             # Central runtime configuration
│   ├── config_parser.py               # Strongly typed configuration loader
│   └── profiles/                      # Persistent user profile adaptation weights
├── deploy/
│   ├── Dockerfile.server              # Multi-stage CUDA 12.2 / TensorRT / JAX / PyTorch Dockerfile
│   ├── docker-compose.yml             # Cloud GPU container orchestration
│   └── run_cloud_server.sh            # Hardware probe & startup launcher
├── docs/
│   └── API_SPECIFICATION.md           # Wire protocol and WebSocket JSON schema documentation
├── src/
│   ├── analytics/
│   │   └── benchmark.py               # CoAdaptationBenchmark, TrialMetrics, ASCII trend graphs
│   ├── audio/
│   │   └── speech_to_text.py          # AudioTranscriberABC, MockTranscriber, WhisperTranscriber
│   ├── hardware/
│   │   └── robot_interface.py         # RobotHardwareABC, MockRobotHardware, ROS2ControlBridge
│   ├── perception/
│   │   ├── hand_tracker.py            # HandTrackerABC & MANO HandPose dataclasses
│   │   ├── mediapipe_tracker.py       # Live CPU-friendly 21-joint MediaPipe tracker
│   │   ├── depth_estimator.py         # DepthEstimatorABC & DepthMap dataclasses
│   │   ├── scene_parser.py            # SceneParserABC, BoundingBox3D, ParsedScene
│   │   └── intent_parser.py           # ParsedIntent, MockLLMIntentParser, StructuredLLMIntentParser
│   ├── simulation/
│   │   ├── simulator.py               # SimulatorABC, SimState, SimAction, ObjectMesh
│   │   └── trajectory_generator.py     # TrajectoryGeneratorABC, AffordanceMap, ForeseenTrajectory
│   ├── policy/
│   │   ├── policy.py                  # PolicyABC, PolicyObservation, PolicyAction
│   │   ├── policy_base.py             # Policy interface compatibility
│   │   ├── neural_policy.py           # NeuralResidualPolicy: real PyTorch MLP, online Reward-Weighted Regression
│   │   ├── discrepancy.py             # DiscrepancyEngine, 112D s_t, EpisodeDiscrepancyReport
│   │   ├── workflow_state.py          # ExecutionPhase enum, WorkflowController state machine
│   │   └── checkpointing.py           # PolicyCheckpointManager for persistent weights
│   ├── safety/
│   │   └── safety_monitor.py          # SafetyMonitor, SafetyStatus, Kinematic Limits & E-Stops
│   ├── transport/
│   │   ├── protocol.py                # FrameMessage, InferenceResponse, serialization
│   │   ├── ws_client.py               # Async WebSocket client
│   │   └── ws_server.py               # Async WebSocket inference backend
│   ├── utils/
│   │   ├── profiler.py                # Thread-safe LatencyProfiler with P99 & ASCII tables
│   │   └── recorder.py                # SessionRecorder for MP4 video & JSONL datasets
│   └── mocks/
│       ├── mock_hand_tracker.py       # Vectorized 21-joint MANO kinematics synthesizer
│       ├── mock_depth_estimator.py     # Vectorized metric depth map generator
│       ├── mock_scene_parser.py       # Intent-conditioned 3D bounding synthesizer
│       ├── mock_affordance_extractor.py # Surface contact probability grounding A(v_i)
│       ├── mock_trajectory_diffusion.py # 60-step foreseen reference trajectory rollout
│       ├── mock_physics_engine.py     # Analytical CPU physics simulator & contacts
│       └── mock_policy.py             # MockResidualPolicy: fixed linear feedback, CPU/test fallback
└── tests/
    ├── test_mocks.py                  # Core mocks and serialization unit tests
    ├── test_mediapipe_tracker.py      # MediaPipe tracker tests
    ├── test_e2e_headless.py           # End-to-end WebSocket streaming tests
    ├── test_phase3.py                 # Affordance, diffusion, and physics tests
    ├── test_policy_discrepancy.py     # 112D state vector, reward, and residual policy tests
    ├── test_profiler_recorder.py      # Latency profiler and session recorder tests
    ├── test_audio_intent.py           # Audio STT and Structured LLM intent tests
    ├── test_workflow_state.py         # Workflow state machine and episode compilation tests
    ├── test_hardware_analytics.py     # Robot hardware, checkpointing, and co-adaptation benchmark tests
    └── test_integration_pipeline.py   # Full E2E pipeline, stress, and safety monitor tests
```

---

## Quickstart & Unified Demo Launcher

Launch the entire client-server system with automated pre-flight checks and background server management using a single command:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run Unified Demo Launcher (Client + Server + GUI HUD)
python apps/run_demo.py --mode mock_remote
```

### Standalone Local Execution (CPU-Only)
```bash
python apps/run_demo.py --mode mock_local
```

---

## Interactive Hotkey Cheat Sheet

When running the visualizer HUD (`apps/local_client.py` or `apps/run_demo.py`), use the following hotkeys:

| Key | Function | Description |
|:---|:---|:---|
| **`ENTER`** or **`c`** | **Step Workflow Phase** | Advance `FORESEEING` $\to$ `WAIT_USER` $\to$ `USER_EXECUTING` $\to$ `ADAPTING` |
| **`h`** | **Expand/Collapse Telemetry** | Toggle between the minimal glance card and the full telemetry dock |
| **`m`** | **Co-Adaptation Panel** | Toggle multi-trial learning curve overlay & error reduction % |
| **`e`** | **Export Benchmark** | Write recorded trials to `logs/benchmarks/benchmark_<ts>.json` and `.csv` |
| **`k`** / `Ctrl+S` | **Save Checkpoint** | Persist learned adaptation weights under `config/profiles/<user_id>/` |
| **`l`** / `Ctrl+L` | **Load Checkpoint** | Restore saved policy weights from disk |
| **`x`** / `Ctrl+R` | **Reset Baseline** | Reset adaptation residual weights to zero baseline |
| **`v`** / `SPACE` | **Voice Push-To-Talk** | Toggle voice speech-to-text listening / transcribing |
| **`g`** | **Voice Guidance** | Mute/unmute spoken workflow instructions on each phase transition |
| **`i`** | **Cycle Intent Prompt** | Cycle target object intent (`remote control`, `coffee cup`, `water bottle`) |
| **`p`** | **Toggle Adaptation** | Enable / pause the online residual policy's learning (Reward-Weighted Regression) |
| **`r`** | **Toggle Recording** | Save MP4 webcam video & JSONL telemetry dataset |
| **`f`** | **Ghost Hand Replay** | Toggle the real-motion afterimage overlay (replays your own recorded attempts) |
| **`t`** | **Toggle Tracker** | Switch between `MEDIAPIPE (LIVE)` and `MOCK (SYNTHETIC)` in real time |
| **`b`** | **3D Bounding Box** | Toggle 3D oriented object bounding wireframe |
| **`d`** | **Depth Inset** | Toggle metric depth picture-in-picture (PIP) inset |
| **`s`** | **Screenshot** | Save timestamped PNG snapshot to disk |
| **`z`** | **Fullscreen** | Toggle the visualizer window between windowed and true fullscreen |
| **`q`** / `ESC` | **Quit** | Gracefully disconnect and exit application |

---

## Safety Guardrails & Interlocks

The system actively enforces the following hard constraints in real time ([`SafetyMonitor`](file:///Users/jameskachamila/Desktop/Precognition/src/safety/safety_monitor.py)):
- **Kinematic Joint Limits**: Clamps joint angles to $[\theta_{\text{min}}, \theta_{\text{max}}]$.
- **Velocity Saturation**: Limits maximum joint velocities ($\Vert \dot{\theta} \Vert \le 2.0\text{ rad/s}$).
- **Workspace Bounds**: Enforces Cartesian volume: $X \in [-0.6, 0.6]\text{ m}, Y \in [-0.5, 0.5]\text{ m}, Z \in [0.1, 0.9]\text{ m}$.
- **Collision Protection**: Monitors distance to scene obstacle hulls; halts execution if clearance $< 2.0\text{ cm}$.
- **Heartbeat Timeout**: Detects telemetry dropout $> 250\text{ ms}$ and triggers an emergency velocity ramp-down.

---

## Automated Test Verification

All 38 unit, integration, and stress tests passed:

```bash
./venv/bin/pytest -v
============================== 38 passed in 4.33s ==============================
```
