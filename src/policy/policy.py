"""Policy interfaces and abstract base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from src.perception.hand_tracker import HandPose
from src.perception.depth_estimator import DepthMap
from src.simulation.simulator import SimState


@dataclass
class PolicyObservation:
    """Multimodal observation supplied to the visuomotor policy."""
    hand_poses: List[HandPose]
    depth_map: Optional[DepthMap] = None
    robot_state: Optional[SimState] = None
    state_vector_112d: Optional[np.ndarray] = None
    timestamp: float = 0.0


@dataclass
class PolicyAction:
    """Action computed by the visuomotor residual adaptation policy."""
    # Joint delta / residual offsets: Delta theta in [-0.08, 0.08] radians
    joint_residuals: np.ndarray
    # Nominal reference joint commands if applicable
    nominal_action: Optional[np.ndarray] = None
    gripper_action: float = 0.0
    confidence: float = 1.0
    reward: float = 0.0

    @property
    def total_action(self) -> np.ndarray:
        if self.nominal_action is not None:
            return self.nominal_action + self.joint_residuals
        return self.joint_residuals


class PolicyABC(ABC):
    """Abstract Base Class for Visuomotor Hand Policies and Residual RL."""

    @abstractmethod
    def evaluate(self, state: np.ndarray) -> PolicyAction:
        """
        Infer residual joint correction action a_t given 112D state vector s_t.

        Args:
            state: (112,) feature vector.

        Returns:
            PolicyAction with joint_residuals in [-0.08, 0.08] radians.
        """
        pass

    @abstractmethod
    def act(self, observation: PolicyObservation) -> PolicyAction:
        """Standard high-level observation inference method."""
        pass

    @abstractmethod
    def update(self, replay_buffer: Any) -> Dict[str, float]:
        """
        Online adaptation learning step (e.g. PPO / SAC policy gradient update).

        Returns:
            Dictionary of training telemetry (loss, mean_reward, kl_divergence, step_count).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset policy hidden state and adaptation buffers."""
        pass
