"""Trajectory planning, affordance grounding, and reference path generator interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from src.perception.hand_tracker import HandPose
from src.perception.scene_parser import BoundingBox3D
from src.simulation.simulator import SimState


@dataclass
class AffordanceMap:
    """Surface contact probability distribution grounded over object geometry."""
    object_label: str
    surface_points: np.ndarray # (N, 3) XYZ coordinates in camera frame
    contact_probabilities: np.ndarray # (N,) array with values in [0, 1]
    hotspots: np.ndarray # (K, 3) primary candidate contact hotspots
    intent: str
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_label": self.object_label,
            "surface_points": self.surface_points.tolist(),
            "contact_probabilities": self.contact_probabilities.tolist(),
            "hotspots": self.hotspots.tolist(),
            "intent": self.intent,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffordanceMap":
        return cls(
            object_label=data["object_label"],
            surface_points=np.array(data["surface_points"], dtype=np.float32),
            contact_probabilities=np.array(data["contact_probabilities"], dtype=np.float32),
            hotspots=np.array(data["hotspots"], dtype=np.float32),
            intent=data.get("intent", ""),
            timestamp=float(data.get("timestamp", 0.0))
        )


@dataclass
class ForeseenWaypoint:
    """A discrete time-indexed waypoint in the 60-step foreseen reference rollout."""
    timestep: int # 1 to 60
    time_offset: float # seconds (e.g. timestep / 30.0)
    hand_keypoints_3d: np.ndarray # (21, 3) in meters
    hand_keypoints_2d: np.ndarray # (21, 2) projected pixels
    wrist_pose: np.ndarray # (6,) [x, y, z, roll, pitch, yaw]
    object_pose: np.ndarray # (6,) [x, y, z, roll, pitch, yaw]
    contact_state: np.ndarray # (5,) contact probability for [thumb, index, middle, ring, pinky]
    gripper_aperture: float = 0.0 # 0.0 = fully open, 1.0 = fully closed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep": self.timestep,
            "time_offset": float(self.time_offset),
            "hand_keypoints_3d": self.hand_keypoints_3d.tolist(),
            "hand_keypoints_2d": self.hand_keypoints_2d.tolist(),
            "wrist_pose": self.wrist_pose.tolist(),
            "object_pose": self.object_pose.tolist(),
            "contact_state": self.contact_state.tolist(),
            "gripper_aperture": float(self.gripper_aperture)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForeseenWaypoint":
        return cls(
            timestep=int(data["timestep"]),
            time_offset=float(data["time_offset"]),
            hand_keypoints_3d=np.array(data["hand_keypoints_3d"], dtype=np.float32),
            hand_keypoints_2d=np.array(data["hand_keypoints_2d"], dtype=np.float32),
            wrist_pose=np.array(data["wrist_pose"], dtype=np.float32),
            object_pose=np.array(data["object_pose"], dtype=np.float32),
            contact_state=np.array(data["contact_state"], dtype=np.float32),
            gripper_aperture=float(data.get("gripper_aperture", 0.0))
        )


@dataclass
class ForeseenTrajectory:
    """Complete 60-step kinematically stable reference rollout tau_ref."""
    intent: str
    target_label: str
    waypoints: List[ForeseenWaypoint] # 60 discrete steps
    duration: float = 2.0 # 60 frames @ 30 FPS = 2.0 seconds

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "target_label": self.target_label,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "duration": float(self.duration)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForeseenTrajectory":
        return cls(
            intent=data.get("intent", ""),
            target_label=data.get("target_label", "target_object"),
            waypoints=[ForeseenWaypoint.from_dict(w) for w in data.get("waypoints", [])],
            duration=float(data.get("duration", 2.0))
        )


@dataclass
class Waypoint:
    """A discrete trajectory waypoint in Cartesian or Joint space."""
    target_pose: np.ndarray # 6D pose or 7D (pos + quat)
    timestamp: float
    joint_targets: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    velocity_limit: float = 1.0


@dataclass
class Trajectory:
    """Parametric or discrete motion trajectory."""
    waypoints: List[Waypoint]
    duration: float

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)


class TrajectoryGeneratorABC(ABC):
    """Abstract Base Class for Diffusion and Optimization trajectory generators."""

    @abstractmethod
    def generate_foreseen_rollout(
        self,
        start_hand_pose: Optional[HandPose],
        target_object: BoundingBox3D,
        affordance_map: AffordanceMap,
        intent: str = "foresee me picking this remote control",
        num_steps: int = 60
    ) -> ForeseenTrajectory:
        """
        Generate a 60-frame kinematically stable foreseen reference trajectory tau_ref.

        Args:
            start_hand_pose: Initial perceived hand pose (or default home position).
            target_object: 3D bounding primitive of target manipuland.
            affordance_map: Surface contact probability hotspots.
            intent: User natural language instruction.
            num_steps: Number of trajectory steps (default 60 = 2.0s @ 30 FPS).

        Returns:
            ForeseenTrajectory containing 60 ForeseenWaypoints.
        """
        pass

    @abstractmethod
    def plan(self, start_state: SimState, goal_pose: np.ndarray) -> Trajectory:
        """Plan a baseline reference joint trajectory."""
        pass
