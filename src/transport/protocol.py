"""Transport message protocol and data serialization utilities."""

import base64
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.perception.hand_tracker import HandPose
from src.perception.scene_parser import ParsedScene
from src.perception.intent_parser import ParsedIntent
from src.simulation.trajectory_generator import AffordanceMap, ForeseenTrajectory
from src.policy.discrepancy import EpisodeDiscrepancyReport
from src.hardware.robot_interface import RobotState
from src.safety.safety_monitor import SafetyStatus


class MessageType(str, Enum):
    FRAME_REQUEST = "frame_request"
    INFERENCE_RESPONSE = "inference_response"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class FrameMessage:
    """Outbound frame sent from local client to remote server."""
    frame_id: int
    client_timestamp: float
    image_base64: str
    width: int
    height: int
    intent: str = "idle"
    workflow_phase: str = "IDLE"
    control_command: Optional[str] = None
    compression: str = "jpeg"
    msg_type: str = MessageType.FRAME_REQUEST.value

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "FrameMessage":
        data = json.loads(raw)
        return cls(
            frame_id=int(data["frame_id"]),
            client_timestamp=float(data["client_timestamp"]),
            image_base64=data["image_base64"],
            width=int(data["width"]),
            height=int(data["height"]),
            intent=data.get("intent", "idle"),
            workflow_phase=data.get("workflow_phase", "IDLE"),
            control_command=data.get("control_command"),
            compression=data.get("compression", "jpeg"),
            msg_type=data.get("msg_type", MessageType.FRAME_REQUEST.value)
        )

    def decode_image(self) -> np.ndarray:
        raw_bytes = base64.b64decode(self.image_base64)
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img


@dataclass
class InferenceResponse:
    """Inference payload returned from server to client with Phase 9 safety & telemetry metrics."""
    frame_id: int
    client_timestamp: float
    server_timestamp: float
    hand_poses: List[dict] = field(default_factory=list)
    depth_heatmap_base64: Optional[str] = None
    # Metric depth, in millimetres, as a 16-bit PNG. The heatmap beside it is a
    # colourmapped picture for the HUD and cannot be inverted back to metres,
    # so reconstructing the room as geometry needs the real numbers.
    depth_raw_base64: Optional[str] = None
    parsed_scene: Optional[dict] = None
    parsed_intent: Optional[dict] = None
    affordance_map: Optional[dict] = None
    foreseen_trajectory: Optional[dict] = None
    policy_residuals: Optional[List[float]] = None
    reward_score: float = 0.0
    discrepancy_norm: float = 0.0
    workflow_phase: str = "IDLE"
    phase_progress: float = 0.0
    episode_report: Optional[dict] = None
    robot_state: Optional[dict] = None
    safety_status: Optional[dict] = None
    benchmark_summary: Optional[dict] = None
    adaptation_status: str = "ACTIVE"
    buffer_step_count: int = 0
    policy_loss: float = 0.0
    # Cumulative RWR gradient steps the policy has taken, so the client can
    # show training happening as a countable, visible event.
    policy_updates: int = 0
    # The accumulated (x, y, z) wrist bias, in metres, that learning has
    # folded into the plan - what the correction arrows on the client render.
    learned_wrist_bias: Optional[List[float]] = None
    gripper_action: float = 0.0
    server_processing_ms: float = 0.0
    msg_type: str = MessageType.INFERENCE_RESPONSE.value

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "InferenceResponse":
        data = json.loads(raw)
        return cls(
            frame_id=int(data["frame_id"]),
            client_timestamp=float(data["client_timestamp"]),
            server_timestamp=float(data["server_timestamp"]),
            hand_poses=data.get("hand_poses", []),
            depth_heatmap_base64=data.get("depth_heatmap_base64"),
            depth_raw_base64=data.get("depth_raw_base64"),
            parsed_scene=data.get("parsed_scene"),
            parsed_intent=data.get("parsed_intent"),
            affordance_map=data.get("affordance_map"),
            foreseen_trajectory=data.get("foreseen_trajectory"),
            policy_residuals=data.get("policy_residuals"),
            reward_score=float(data.get("reward_score", 0.0)),
            discrepancy_norm=float(data.get("discrepancy_norm", 0.0)),
            workflow_phase=data.get("workflow_phase", "IDLE"),
            phase_progress=float(data.get("phase_progress", 0.0)),
            episode_report=data.get("episode_report"),
            robot_state=data.get("robot_state"),
            safety_status=data.get("safety_status"),
            benchmark_summary=data.get("benchmark_summary"),
            adaptation_status=data.get("adaptation_status", "ACTIVE"),
            buffer_step_count=int(data.get("buffer_step_count", 0)),
            policy_loss=float(data.get("policy_loss", 0.0)),
            policy_updates=int(data.get("policy_updates", 0)),
            learned_wrist_bias=data.get("learned_wrist_bias"),
            gripper_action=float(data.get("gripper_action", 0.0)),
            server_processing_ms=float(data.get("server_processing_ms", 0.0)),
            msg_type=data.get("msg_type", MessageType.INFERENCE_RESPONSE.value)
        )

    def get_hand_poses(self) -> List[HandPose]:
        return [HandPose.from_dict(hp) for hp in self.hand_poses]

    def get_parsed_scene(self) -> Optional[ParsedScene]:
        if self.parsed_scene is None:
            return None
        return ParsedScene.from_dict(self.parsed_scene)

    def get_parsed_intent(self) -> Optional[ParsedIntent]:
        if self.parsed_intent is None:
            return None
        return ParsedIntent.from_dict(self.parsed_intent)

    def get_affordance_map(self) -> Optional[AffordanceMap]:
        if self.affordance_map is None:
            return None
        return AffordanceMap.from_dict(self.affordance_map)

    def get_foreseen_trajectory(self) -> Optional[ForeseenTrajectory]:
        if self.foreseen_trajectory is None:
            return None
        return ForeseenTrajectory.from_dict(self.foreseen_trajectory)

    def get_episode_report(self) -> Optional[EpisodeDiscrepancyReport]:
        if self.episode_report is None:
            return None
        return EpisodeDiscrepancyReport.from_dict(self.episode_report)

    def get_robot_state(self) -> Optional[RobotState]:
        if self.robot_state is None:
            return None
        return RobotState.from_dict(self.robot_state)

    def get_safety_status(self) -> Optional[SafetyStatus]:
        if self.safety_status is None:
            return None
        return SafetyStatus(
            is_safe=bool(self.safety_status.get("is_safe", True)),
            is_e_stopped=bool(self.safety_status.get("is_e_stopped", False)),
            warning_flags=list(self.safety_status.get("warning_flags", [])),
            clamped_joint_positions=np.array(self.safety_status.get("clamped_joint_positions", np.zeros(7)), dtype=np.float32),
            clamped_joint_velocities=np.array(self.safety_status.get("clamped_joint_velocities", np.zeros(7)), dtype=np.float32),
            min_obstacle_clearance_meters=float(self.safety_status.get("min_obstacle_clearance_meters", 1.0)),
            heartbeat_latency_ms=float(self.safety_status.get("heartbeat_latency_ms", 0.0)),
            timestamp=float(self.safety_status.get("timestamp", time.time()))
        )

    def decode_depth_raw(self) -> Optional[np.ndarray]:
        """Metric depth in metres, or None if the server did not send any."""
        if not self.depth_raw_base64:
            return None
        raw = np.frombuffer(base64.b64decode(self.depth_raw_base64), dtype=np.uint8)
        mm = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if mm is None:
            return None
        return (mm.astype(np.float32) / 1000.0)

    def decode_depth_heatmap(self) -> Optional[np.ndarray]:
        if not self.depth_heatmap_base64:
            return None
        raw_bytes = base64.b64decode(self.depth_heatmap_base64)
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_depth_to_base64(depth_m: np.ndarray, max_width: int = 160) -> Optional[str]:
    """Metric depth -> base64 16-bit PNG in millimetres.

    Downsampled first: the client rebuilds the scene on a lattice of well under
    a hundred cells across, so sending full resolution would be spending
    bandwidth on detail that is averaged away on arrival.
    """
    if depth_m is None or depth_m.size == 0:
        return None
    d = np.asarray(depth_m, dtype=np.float32)
    h, w = d.shape[:2]
    if w > max_width:
        d = cv2.resize(d, (max_width, max(2, int(round(h * max_width / w)))),
                       interpolation=cv2.INTER_NEAREST)
    mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", mm)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None


def encode_image_to_base64(image: np.ndarray, quality: int = 80) -> str:
    """Compress BGR image to JPEG and return base64 string."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, buffer = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return base64.b64encode(buffer).decode("utf-8")
