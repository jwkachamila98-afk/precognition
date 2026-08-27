"""Physics simulation interfaces and state representations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ObjectMesh:
    """Procedural or CAD mesh definition for simulated objects."""
    name: str
    vertices: np.ndarray # (V, 3) vertex coordinates in local frame
    faces: Optional[np.ndarray] = None # (F, 3) triangle indices
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32)) # Euler [r, p, y]
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vertices": self.vertices.tolist(),
            "faces": self.faces.tolist() if self.faces is not None else None,
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "scale": self.scale.tolist()
        }


@dataclass
class SimState:
    """Robot and environment simulation state."""
    # Joint configuration (generalized positions q)
    joint_positions: np.ndarray
    # Joint velocities (q_dot)
    joint_velocities: np.ndarray
    # End-effector 6D pose [x, y, z, roll, pitch, yaw] or 7D [x, y, z, qx, qy, qz, qw]
    ee_pose: np.ndarray
    # Contact normal forces / torques on fingertips/gripper (6,)
    contact_forces: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    # Active object poses in environment: dict of name -> (6,) pose
    object_poses: Dict[str, np.ndarray] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": self.joint_velocities.tolist(),
            "ee_pose": self.ee_pose.tolist(),
            "contact_forces": self.contact_forces.tolist(),
            "object_poses": {k: v.tolist() for k, v in self.object_poses.items()},
            "timestamp": self.timestamp
        }


@dataclass
class SimAction:
    """Action sent to the simulator or robot controller."""
    # Target joint angles (position control)
    target_joint_positions: Optional[np.ndarray] = None
    # Joint feed-forward torques (torque control)
    joint_torques: Optional[np.ndarray] = None
    # Gripper or multi-finger aperture [0.0 = fully open, 1.0 = fully closed]
    gripper_command: float = 0.0


class SimulatorABC(ABC):
    """Abstract Base Class for Physics Simulation Engines (MuJoCo / MJX / IsaacGym / MockPhysicsEngine)."""

    @abstractmethod
    def instantiate_object_mesh(
        self,
        mesh_name: str,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None
    ) -> int:
        """
        Instantiate an object collision/visual mesh in the physics simulation.

        Args:
            mesh_name: Identifier for procedural or asset mesh (e.g. 'remote_control', 'mug').
            position: Initial 3D Cartesian position [x, y, z] in world/camera frame.
            orientation: Initial Euler angles [roll, pitch, yaw].
            scale: 3D scale factors [sx, sy, sz].

        Returns:
            Object instance ID integer.
        """
        pass

    @abstractmethod
    def reset(self) -> SimState:
        """Reset the simulation to initial configuration and return state."""
        pass

    @abstractmethod
    def step(self, action: SimAction) -> SimState:
        """
        Advance simulation physics by one timestep dt.

        Args:
            action: Control commands for joints and actuators.

        Returns:
            Updated SimState.
        """
        pass

    @abstractmethod
    def get_state(self) -> SimState:
        """Get current physical simulator state."""
        pass

    @abstractmethod
    def set_state(self, state: SimState) -> None:
        """Teleport/synchronize simulator to an external state."""
        pass
