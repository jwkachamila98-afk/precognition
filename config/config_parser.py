"""Configuration parser and dataclasses for Visuomotor Hand Policy Architecture."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal
import yaml


@dataclass
class SystemSettings:
    mode: str = "mock_local" # 'mock_local' | 'mock_remote' (also accepts 'mock' | 'remote')
    intent: str = "foresee me picking this remote control"
    log_level: str = "INFO"
    seed: int = 42

    def get_normalized_mode(self) -> str:
        """Normalize mode string to 'mock_local' or 'mock_remote'."""
        if self.mode in ("mock", "mock_local"):
            return "mock_local"
        if self.mode in ("remote", "mock_remote"):
            return "mock_remote"
        return self.mode


@dataclass
class CameraSettings:
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    use_synthetic_if_unavailable: bool = True


@dataclass
class NetworkSettings:
    server_host: str = "127.0.0.1"
    server_port: int = 8765
    compression_format: str = "jpeg"
    compression_quality: int = 80
    timeout_seconds: float = 5.0
    reconnect_interval_seconds: float = 2.0


@dataclass
class HandTrackerSettings:
    tracker_type: Literal["mediapipe", "mock"] = "mediapipe"
    max_hands: int = 2
    confidence_threshold: float = 0.5
    mano_pose_dims: int = 48
    mano_shape_dims: int = 10
    num_keypoints: int = 21


@dataclass
class DepthEstimatorSettings:
    min_depth_meters: float = 0.2
    max_depth_meters: float = 2.5
    output_width: int = 320
    output_height: int = 240


@dataclass
class SceneParserSettings:
    num_points: int = 400


@dataclass
class PerceptionSettings:
    hand_tracker: HandTrackerSettings = field(default_factory=HandTrackerSettings)
    depth_estimator: DepthEstimatorSettings = field(default_factory=DepthEstimatorSettings)
    scene_parser: SceneParserSettings = field(default_factory=SceneParserSettings)


@dataclass
class VisualizationSettings:
    window_name: str = "Visuomotor Hand Policy - Local Client"
    show_fps: bool = True
    draw_skeleton: bool = True
    draw_depth_inset: bool = True
    draw_bounding_box: bool = True
    depth_inset_scale: float = 0.28
    rerun_enabled: bool = False


@dataclass
class AppConfig:
    system: SystemSettings = field(default_factory=SystemSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    perception: PerceptionSettings = field(default_factory=PerceptionSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        return cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Construct AppConfig from a dictionary with nested dataclasses."""
        sys_data = data.get("system", {})
        cam_data = data.get("camera", {})
        net_data = data.get("network", {})
        percep_data = data.get("perception", {})
        vis_data = data.get("visualization", {})

        ht_data = percep_data.get("hand_tracker", {})
        de_data = percep_data.get("depth_estimator", {})
        sp_data = percep_data.get("scene_parser", {})

        return cls(
            system=SystemSettings(**sys_data),
            camera=CameraSettings(**cam_data),
            network=NetworkSettings(**net_data),
            perception=PerceptionSettings(
                hand_tracker=HandTrackerSettings(**ht_data),
                depth_estimator=DepthEstimatorSettings(**de_data),
                scene_parser=SceneParserSettings(**sp_data),
            ),
            visualization=VisualizationSettings(**vis_data),
        )
