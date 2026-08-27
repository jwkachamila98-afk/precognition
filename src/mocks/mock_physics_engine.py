"""Lightweight analytical CPU physics simulation engine mock."""

import time
from typing import Dict, List, Optional
import numpy as np

from src.simulation.simulator import ObjectMesh, SimAction, SimState, SimulatorABC


class MockPhysicsEngine(SimulatorABC):
    """
    Lightweight CPU physics simulator engine.
    Accepts procedural object definitions, tracks hand-object joint trajectories,
    and calculates analytical contact state vectors without requiring heavy JAX/MuJoCo/CUDA dependencies.
    """

    def __init__(self, num_dof: int = 7, dt: float = 0.01) -> None:
        self.num_dof = num_dof
        self.dt = dt
        self._q = np.zeros(num_dof, dtype=np.float32)
        self._q_dot = np.zeros(num_dof, dtype=np.float32)
        self._ee_pose = np.array([0.08, 0.08, 0.48, 0.0, 0.0, 0.0], dtype=np.float32)
        self._contact_forces = np.zeros(6, dtype=np.float32)
        self._objects: Dict[int, ObjectMesh] = {}
        self._next_obj_id = 1
        self._timestamp = time.time()

    def instantiate_object_mesh(
        self,
        mesh_name: str,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None
    ) -> int:
        """Instantiate a procedural object collision primitive."""
        obj_id = self._next_obj_id
        self._next_obj_id += 1

        # Generate canonical box vertices if procedural
        s = scale if scale is not None else np.array([0.08, 0.12, 0.06], dtype=np.float32)
        dx, dy, dz = s / 2.0
        canonical_verts = np.array([
            [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
            [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz]
        ], dtype=np.float32)

        mesh = ObjectMesh(
            name=mesh_name,
            vertices=canonical_verts,
            position=position.astype(np.float32),
            orientation=orientation.astype(np.float32) if orientation is not None else np.zeros(3, dtype=np.float32),
            scale=s
        )
        self._objects[obj_id] = mesh
        return obj_id

    def reset(self) -> SimState:
        self._q.fill(0.0)
        self._q_dot.fill(0.0)
        self._ee_pose = np.array([0.08, 0.08, 0.48, 0.0, 0.0, 0.0], dtype=np.float32)
        self._contact_forces.fill(0.0)
        self._timestamp = time.time()
        return self.get_state()

    def step(self, action: SimAction) -> SimState:
        self._timestamp += self.dt

        # Integrate joint dynamics
        if action.target_joint_positions is not None:
            target = action.target_joint_positions[:self.num_dof]
            error = target - self._q
            self._q_dot = error * 8.0 # Proportional velocity control
            self._q += self._q_dot * self.dt

        # Mock forward kinematics for end-effector
        self._ee_pose[0] = 0.08 + 0.05 * float(np.sin(self._q[0]))
        self._ee_pose[1] = 0.08 + 0.03 * float(np.cos(self._q[1]))
        self._ee_pose[2] = 0.48 + 0.04 * float(np.sin(self._q[2]))

        # Calculate analytical contact state against registered objects
        self._contact_forces.fill(0.0)
        for obj in self._objects.values():
            dist = np.linalg.norm(self._ee_pose[:3] - obj.position)
            # Threshold distance for surface contact (~5 cm)
            if dist < 0.05 and action.gripper_command > 0.3:
                # Normal force proportional to grasp closure and proximity
                f_norm = float(15.0 * action.gripper_command * (1.0 - dist / 0.05))
                self._contact_forces[2] = f_norm # Z normal force
                self._contact_forces[0] = f_norm * 0.2 # Friction shear
                self._contact_forces[1] = f_norm * 0.2

                # If firmly gripped, object moves with end effector
                if action.gripper_command > 0.7:
                    obj.position = self._ee_pose[:3].copy()

        return self.get_state()

    def get_state(self) -> SimState:
        obj_poses = {f"obj_{k}_{v.name}": np.concatenate([v.position, v.orientation]) for k, v in self._objects.items()}
        return SimState(
            joint_positions=self._q.copy(),
            joint_velocities=self._q_dot.copy(),
            ee_pose=self._ee_pose.copy(),
            contact_forces=self._contact_forces.copy(),
            object_poses=obj_poses,
            timestamp=self._timestamp
        )

    def set_state(self, state: SimState) -> None:
        self._q = state.joint_positions.copy()
        self._q_dot = state.joint_velocities.copy()
        self._ee_pose = state.ee_pose.copy()
        self._contact_forces = state.contact_forces.copy()
        self._timestamp = state.timestamp


# Backwards compatibility alias
MockSimulator = MockPhysicsEngine
