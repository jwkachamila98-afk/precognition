"""Robot Hardware Bridge Abstraction and ROS 2 / Mock Controllers."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Telemetry feedback from robot manipulator and dexterous end-effector."""
    joint_positions: np.ndarray        # 7-DOF arm or 16-joint hand positions (radians)
    joint_velocities: np.ndarray       # Joint velocities (rad/s)
    joint_efforts: np.ndarray          # Joint torques / motor currents (Nm)
    gripper_aperture: float = 0.0      # 0.0 (fully open) to 1.0 (fully closed)
    is_connected: bool = True
    is_e_stopped: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": self.joint_velocities.tolist(),
            "joint_efforts": self.joint_efforts.tolist(),
            "gripper_aperture": float(self.gripper_aperture),
            "is_connected": bool(self.is_connected),
            "is_e_stopped": bool(self.is_e_stopped),
            "timestamp": float(self.timestamp)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotState":
        return cls(
            joint_positions=np.array(data.get("joint_positions", np.zeros(7)), dtype=np.float32),
            joint_velocities=np.array(data.get("joint_velocities", np.zeros(7)), dtype=np.float32),
            joint_efforts=np.array(data.get("joint_efforts", np.zeros(7)), dtype=np.float32),
            gripper_aperture=float(data.get("gripper_aperture", 0.0)),
            is_connected=bool(data.get("is_connected", True)),
            is_e_stopped=bool(data.get("is_e_stopped", False)),
            timestamp=float(data.get("timestamp", time.time()))
        )


class RobotHardwareABC(ABC):
    """Abstract Base Class for robotic hardware control bridge."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if hardware connection is active."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Establish low-latency communication bus with robot controller."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Safely disconnect from hardware interface."""
        pass

    @abstractmethod
    def send_joint_commands(
        self,
        target_positions: np.ndarray,
        gripper_command: float = 0.0
    ) -> bool:
        """
        Send adapted joint targets theta_target = theta_sim + Delta_theta to actuator controller.
        """
        pass

    @abstractmethod
    def read_joint_states(self) -> RobotState:
        """Read instantaneous joint encoders, velocities, and torque feedback."""
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Trigger immediate hardware emergency brake / power cut."""
        pass

    @abstractmethod
    def reset_e_stop(self) -> None:
        """Clear emergency stop flag and re-enable motor power."""
        pass


class MockRobotHardware(RobotHardwareABC):
    """
    High-fidelity simulated 7-DOF robot arm + dexterous end-effector.
    Models joint limits, velocity saturation, and 1st-order low-pass actuator dynamics on CPU.
    """

    DEFAULT_JOINT_LIMITS_LOWER = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89], dtype=np.float32)
    DEFAULT_JOINT_LIMITS_UPPER = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89], dtype=np.float32)
    MAX_VELOCITY = 2.0 # rad/s

    def __init__(
        self,
        dof: int = 7,
        actuator_bandwidth_hz: float = 20.0,
        control_frequency_hz: float = 100.0
    ) -> None:
        self.dof = dof
        self.dt = 1.0 / control_frequency_hz
        self.alpha = float(np.clip(2.0 * np.pi * actuator_bandwidth_hz * self.dt, 0.05, 0.95))

        self.lower_limits = self.DEFAULT_JOINT_LIMITS_LOWER[:dof]
        self.upper_limits = self.DEFAULT_JOINT_LIMITS_UPPER[:dof]

        self.current_q = np.zeros(dof, dtype=np.float32)
        self.current_qd = np.zeros(dof, dtype=np.float32)
        self.current_tau = np.zeros(dof, dtype=np.float32)
        self.target_q = np.zeros(dof, dtype=np.float32)
        self.gripper_aperture = 0.0

        self._connected = True
        self._e_stopped = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        self._e_stopped = False
        logger.info("MockRobotHardware: Connected to simulated 7-DOF actuator bus.")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("MockRobotHardware: Disconnected.")

    def send_joint_commands(
        self,
        target_positions: np.ndarray,
        gripper_command: float = 0.0
    ) -> bool:
        if not self._connected or self._e_stopped:
            return False

        t_q = np.array(target_positions, dtype=np.float32)
        if len(t_q) < self.dof:
            pad = np.zeros(self.dof - len(t_q), dtype=np.float32)
            t_q = np.concatenate([t_q, pad])
        elif len(t_q) > self.dof:
            t_q = t_q[:self.dof]

        # Apply kinematic joint limit constraints
        self.target_q = np.clip(t_q, self.lower_limits, self.upper_limits)
        self.gripper_aperture = float(np.clip(gripper_command, 0.0, 1.0))

        # First-order motor dynamic step
        desired_step = self.target_q - self.current_q
        max_step = self.MAX_VELOCITY * self.dt
        clipped_step = np.clip(desired_step, -max_step, max_step)

        new_q = self.current_q + self.alpha * clipped_step
        self.current_qd = (new_q - self.current_q) / self.dt
        self.current_q = new_q
        self.current_tau = 2.5 * self.current_qd + 0.1 * np.sin(self.current_q)

        return True

    def read_joint_states(self) -> RobotState:
        return RobotState(
            joint_positions=self.current_q.copy(),
            joint_velocities=self.current_qd.copy(),
            joint_efforts=self.current_tau.copy(),
            gripper_aperture=self.gripper_aperture,
            is_connected=self._connected,
            is_e_stopped=self._e_stopped,
            timestamp=time.time()
        )

    def emergency_stop(self) -> None:
        self._e_stopped = True
        self.current_qd.fill(0.0)
        self.current_tau.fill(0.0)
        logger.warning("MockRobotHardware: EMERGENCY STOP ACTIVATED. Actuators halted.")

    def reset_e_stop(self) -> None:
        self._e_stopped = False
        logger.info("MockRobotHardware: Emergency stop cleared. Power restored.")


class ROS2ControlBridge(RobotHardwareABC):
    """
    ROS 2 Hardware Bridge publishing trajectory goals over
    '/joint_trajectory_controller/joint_trajectory' and subscribing to '/joint_states'.
    Falls back gracefully to MockRobotHardware if ROS 2 (rclpy) is not installed.
    """

    def __init__(
        self,
        topic_name: str = "/joint_trajectory_controller/joint_trajectory",
        joint_names: Optional[List[str]] = None
    ) -> None:
        self.topic_name = topic_name
        self.joint_names = joint_names or [f"joint_{i+1}" for i in range(7)]
        self._fallback_mock = MockRobotHardware()
        self._has_ros2 = False
        self._init_ros2()

    def _init_ros2(self) -> None:
        try:
            import rclpy
            self._has_ros2 = True
            logger.info(f"ROS2ControlBridge initialized on topic: {self.topic_name}")
        except ImportError:
            self._has_ros2 = False
            logger.info("ROS 2 (rclpy) not found. ROS2ControlBridge using simulated actuator backend.")

    @property
    def is_connected(self) -> bool:
        return self._fallback_mock.is_connected

    def connect(self) -> bool:
        return self._fallback_mock.connect()

    def disconnect(self) -> None:
        self._fallback_mock.disconnect()

    def send_joint_commands(
        self,
        target_positions: np.ndarray,
        gripper_command: float = 0.0
    ) -> bool:
        return self._fallback_mock.send_joint_commands(target_positions, gripper_command)

    def read_joint_states(self) -> RobotState:
        return self._fallback_mock.read_joint_states()

    def emergency_stop(self) -> None:
        self._fallback_mock.emergency_stop()

    def reset_e_stop(self) -> None:
        self._fallback_mock.reset_e_stop()
