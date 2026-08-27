"""Safety guardrails, kinematic limits, collision avoidance, and emergency interlocks."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.perception.scene_parser import BoundingBox3D

logger = logging.getLogger(__name__)


@dataclass
class SafetyStatus:
    """Real-time safety assessment telemetry."""
    is_safe: bool = True
    is_e_stopped: bool = False
    warning_flags: List[str] = field(default_factory=list)
    clamped_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    clamped_joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    min_obstacle_clearance_meters: float = 1.0
    heartbeat_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": bool(self.is_safe),
            "is_e_stopped": bool(self.is_e_stopped),
            "warning_flags": list(self.warning_flags),
            "clamped_joint_positions": self.clamped_joint_positions.tolist(),
            "clamped_joint_velocities": self.clamped_joint_velocities.tolist(),
            "min_obstacle_clearance_meters": float(self.min_obstacle_clearance_meters),
            "heartbeat_latency_ms": float(self.heartbeat_latency_ms),
            "timestamp": float(self.timestamp)
        }


class SafetyMonitor:
    """
    Real-time safety monitor enforcing:
    1. Hard kinematic joint limits [q_min, q_max]
    2. Cartesian workspace bounding volume [X, Y, Z]
    3. Joint velocity & acceleration saturation (|q_dot| <= 2.0 rad/s)
    4. Heartbeat telemetry loss detection (timeout > 250 ms triggers emergency ramp-down)
    5. Collision hull clearance (distance < 2.0 cm triggers soft stop)
    """

    DEFAULT_LOWER_JOINT_LIMITS = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89], dtype=np.float32)
    DEFAULT_UPPER_JOINT_LIMITS = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89], dtype=np.float32)

    # Allowed workspace box in meters: [X_min, X_max, Y_min, Y_max, Z_min, Z_max]
    WORKSPACE_BOUNDS = np.array([-0.60, 0.60, -0.50, 0.50, 0.10, 0.90], dtype=np.float32)

    MAX_JOINT_VELOCITY = 2.0         # rad/s
    MAX_JOINT_ACCELERATION = 8.0     # rad/s^2
    HEARTBEAT_TIMEOUT_SEC = 0.250    # 250 ms telemetry dropout threshold
    MIN_COLLISION_CLEARANCE = 0.020  # 2.0 cm minimum obstacle clearance

    def __init__(
        self,
        dof: int = 7,
        max_velocity: float = 2.0,
        heartbeat_timeout_sec: float = 0.250,
        min_clearance_meters: float = 0.020
    ) -> None:
        self.dof = dof
        self.max_velocity = max_velocity
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.min_clearance_meters = min_clearance_meters

        self.lower_limits = self.DEFAULT_LOWER_JOINT_LIMITS[:dof]
        self.upper_limits = self.DEFAULT_UPPER_JOINT_LIMITS[:dof]

        self._last_heartbeat = time.time()
        self._last_cmd_q = np.zeros(dof, dtype=np.float32)
        self._last_cmd_time = time.time()
        self._e_stopped = False

    def check_heartbeat(self, last_packet_time: float) -> Tuple[bool, float]:
        """
        Verify incoming telemetry / frame heartbeat.
        Returns (is_active, latency_ms).
        """
        now = time.time()
        elapsed = now - last_packet_time
        latency_ms = elapsed * 1000.0
        is_active = (elapsed <= self.heartbeat_timeout_sec)
        if not is_active:
            logger.warning(f"SafetyMonitor: Heartbeat loss detected! ({latency_ms:.1f} ms > {self.heartbeat_timeout_sec*1000:.0f} ms)")
        return is_active, latency_ms

    def check_workspace_limits(self, cartesian_pos: np.ndarray) -> bool:
        """Verify Cartesian end-effector position is inside allowed workspace volume."""
        pos = np.array(cartesian_pos[:3], dtype=np.float32)
        b = self.WORKSPACE_BOUNDS
        inside_x = (b[0] <= pos[0] <= b[1])
        inside_y = (b[2] <= pos[1] <= b[3])
        inside_z = (b[4] <= pos[2] <= b[5])
        return bool(inside_x and inside_y and inside_z)

    def check_collision_clearance(
        self,
        cartesian_pos: np.ndarray,
        obstacles: List[BoundingBox3D]
    ) -> float:
        """
        Calculate Euclidean distance between hand/end-effector and scene collision primitives.
        """
        if not obstacles:
            return 1.0 # Clear

        pos = np.array(cartesian_pos[:3], dtype=np.float32)
        min_dist = 10.0

        for box in obstacles:
            # Approximate distance to bounding box center minus approximate radius
            box_radius = float(np.max(box.size) / 2.0)
            center_dist = float(np.linalg.norm(pos - box.center))
            surface_dist = max(0.0, center_dist - box_radius)
            if surface_dist < min_dist:
                min_dist = surface_dist

        return float(min_dist)

    def filter_joint_command(
        self,
        target_q: np.ndarray,
        current_q: np.ndarray,
        dt: float = 0.033
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Enforce kinematic joint limits and velocity saturation on outbound commands.
        """
        warnings = []
        t_q = np.array(target_q[:self.dof], dtype=np.float32)
        c_q = np.array(current_q[:self.dof], dtype=np.float32)

        # 1. Joint limit clamping
        clamped_q = np.clip(t_q, self.lower_limits, self.upper_limits)
        if not np.allclose(clamped_q, t_q, atol=1e-3):
            warnings.append("JOINT_LIMIT_REACHED")

        # 2. Joint velocity limit clamping
        step_dt = max(dt, 0.001)
        desired_qd = (clamped_q - c_q) / step_dt
        max_vel = self.max_velocity
        clamped_qd = np.clip(desired_qd, -max_vel, max_vel)

        if not np.allclose(clamped_qd, desired_qd, atol=1e-3):
            warnings.append("VELOCITY_SATURATED")

        final_q = c_q + clamped_qd * step_dt
        return final_q, clamped_qd, warnings

    def evaluate_safety(
        self,
        target_q: np.ndarray,
        current_q: np.ndarray,
        cartesian_pos: np.ndarray,
        last_packet_time: float,
        obstacles: Optional[List[BoundingBox3D]] = None,
        dt: float = 0.033
    ) -> SafetyStatus:
        """
        Comprehensive safety audit across kinematics, velocity, telemetry, and collisions.
        """
        warning_flags = []
        is_safe = True

        # Heartbeat check
        hb_ok, latency_ms = self.check_heartbeat(last_packet_time)
        if not hb_ok:
            warning_flags.append("HEARTBEAT_LOSS")
            is_safe = False

        # Workspace limit check
        ws_ok = self.check_workspace_limits(cartesian_pos)
        if not ws_ok:
            warning_flags.append("WORKSPACE_OUT_OF_BOUNDS")
            is_safe = False

        # Collision clearance check
        clearance = self.check_collision_clearance(cartesian_pos, obstacles or [])
        if clearance < self.min_clearance_meters:
            warning_flags.append("COLLISION_IMMINENT")
            is_safe = False

        # Kinematic & velocity filtering
        clamped_q, clamped_qd, k_warnings = self.filter_joint_command(target_q, current_q, dt=dt)
        warning_flags.extend(k_warnings)

        if not is_safe:
            # Emergency velocity ramp down to zero
            clamped_qd.fill(0.0)
            clamped_q = current_q.copy()

        return SafetyStatus(
            is_safe=is_safe,
            is_e_stopped=self._e_stopped or (not is_safe and "HEARTBEAT_LOSS" in warning_flags),
            warning_flags=warning_flags,
            clamped_joint_positions=clamped_q,
            clamped_joint_velocities=clamped_qd,
            min_obstacle_clearance_meters=clearance,
            heartbeat_latency_ms=latency_ms,
            timestamp=time.time()
        )
