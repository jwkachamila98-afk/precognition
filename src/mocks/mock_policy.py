"""Synthetic Visuomotor Residual Adaptation Policy Mock (Phase 4)."""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.policy.policy import PolicyABC, PolicyAction, PolicyObservation


class MockResidualPolicy(PolicyABC):
    """
    Lightweight CPU-friendly Residual Adaptation Policy.
    Accepts 112D state vector s_t, outputs residual joint correction deltas
    a_t = Delta theta_t in [-0.08, 0.08] radians, and simulates an online learning
    update loop (mock PPO/SAC gradient step) to minimize tracking discrepancy over time.
    """

    def __init__(
        self,
        state_dim: int = 112,
        action_dim: int = 7,
        learning_rate: float = 3e-4,
        max_residual: float = 0.08
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.max_residual = max_residual

        # Internal policy linear weights & bias for fast CPU inference W @ s + b
        np.random.seed(42)
        self.weights = np.random.uniform(-0.01, 0.01, size=(action_dim, state_dim)).astype(np.float32)
        # Emphasize proportional feedback on the first 7 keypoint discrepancy features
        for i in range(action_dim):
            self.weights[i, i * 3] = -0.35

        self.bias = np.zeros(action_dim, dtype=np.float32)

        # Online adaptation buffers & telemetry
        self.adaptation_active = True
        self.step_count = 0
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_capacity = 120
        self.loss_history: List[float] = [0.05]
        self.reward_history: List[float] = [0.75]
        self.cumulative_adaptations = 0

    @property
    def W(self) -> np.ndarray:
        return self.weights

    @W.setter
    def W(self, val: np.ndarray) -> None:
        self.weights = np.array(val, dtype=np.float32)

    @property
    def b(self) -> np.ndarray:
        return self.bias

    @b.setter
    def b(self, val: np.ndarray) -> None:
        self.bias = np.array(val, dtype=np.float32)

    def evaluate(self, state: np.ndarray) -> PolicyAction:
        """
        Infer residual action vector a_t given 112D state vector s_t.
        """
        if len(state) < self.state_dim:
            s_padded = np.zeros(self.state_dim, dtype=np.float32)
            s_padded[:len(state)] = state
            state = s_padded

        # Linear network inference with tanh saturation bounded to [-max_residual, +max_residual]
        raw_logits = self.weights @ state + self.bias
        joint_residuals = np.tanh(raw_logits) * self.max_residual

        # Gripper action based on state progress feature at index 99
        progress = state[99] if len(state) > 99 else 0.0
        gripper_cmd = float(np.clip((progress - 0.4) / 0.3, 0.0, 1.0))

        return PolicyAction(
            joint_residuals=joint_residuals.astype(np.float32),
            gripper_action=gripper_cmd,
            confidence=0.96
        )

    def act(self, observation: PolicyObservation) -> PolicyAction:
        """High-level observation handler."""
        if observation.state_vector_112d is not None:
            return self.evaluate(observation.state_vector_112d)

        dummy_state = np.zeros(self.state_dim, dtype=np.float32)
        if observation.hand_poses:
            p = observation.hand_poses[0]
            dummy_state[63:66] = p.keypoints_3d[0]
        return self.evaluate(dummy_state)

    def record_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: Optional[np.ndarray] = None
    ) -> None:
        """Append (s, a, r, s') transition to online replay buffer."""
        if not self.adaptation_active:
            return

        self.step_count += 1
        self.buffer.append({
            "state": state.copy(),
            "action": action.copy(),
            "reward": float(reward),
            "next_state": next_state.copy() if next_state is not None else None,
            "timestamp": time.time()
        })

        if len(self.buffer) > self.buffer_capacity:
            self.buffer.pop(0)

        # Automatically trigger gradient step every 30 online steps
        if self.step_count % 30 == 0:
            self.update(self.buffer)

    def update(self, replay_buffer: Any) -> Dict[str, float]:
        """
        Simulate an online PPO policy gradient update step.
        Adjusts weights to minimize tracking discrepancy and maximize reward.
        """
        if not self.buffer:
            return {
                "loss": 0.0,
                "mean_reward": 0.0,
                "learning_rate": self.learning_rate,
                "step_count": self.step_count
            }

        recent_rewards = [t["reward"] for t in self.buffer[-30:]]
        mean_r = float(np.mean(recent_rewards))
        self.reward_history.append(mean_r)

        current_loss = max(0.001, float(1.0 - mean_r + 0.02 * np.random.randn()))
        self.loss_history.append(current_loss)

        grad_direction = -0.001 * np.mean([t["state"] for t in self.buffer[-10:]], axis=0)
        for i in range(self.action_dim):
            self.weights[i] += self.learning_rate * grad_direction

        self.cumulative_adaptations += 1

        return {
            "loss": current_loss,
            "mean_reward": mean_r,
            "learning_rate": self.learning_rate,
            "step_count": self.step_count,
            "cumulative_updates": self.cumulative_adaptations
        }

    def reset(self) -> None:
        self.buffer.clear()
        self.step_count = 0


# Backwards compatibility alias
MockPolicy = MockResidualPolicy
