"""Policy checkpointing and user-specific adaptation profile storage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PolicyCheckpointManager:
    """
    Manages saving, loading, and persistence of learned residual policy network weights (pi_phi)
    and user motion profiles under config/profiles/<user_id>/.
    Ensures learned adaptations persist across application and server restarts.
    """

    def __init__(self, base_profiles_dir: str = "config/profiles") -> None:
        self.base_dir = Path(base_profiles_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_user_dir(self, user_id: str) -> Path:
        user_dir = self.base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def save_checkpoint(
        self,
        policy: Any,
        user_id: str = "default_user",
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """
        Extract weights, biases, and optimizer state from residual policy and save to JSON.
        """
        user_dir = self._get_user_dir(user_id)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{checkpoint_name or 'policy_checkpoint'}_{ts}.json"
        save_path = user_dir / filename

        # Extract weights from MockResidualPolicy or PyTorch Policy
        weights_dict: Dict[str, Any] = {}
        if hasattr(policy, "W") and hasattr(policy, "b"):
            weights_dict["W"] = policy.W.tolist() if isinstance(policy.W, np.ndarray) else policy.W
            weights_dict["b"] = policy.b.tolist() if isinstance(policy.b, np.ndarray) else policy.b
        elif hasattr(policy, "state_dict"):
            # PyTorch Module state dict
            state = policy.state_dict()
            weights_dict = {k: v.cpu().numpy().tolist() for k, v in state.items()}

        payload = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "step_count": getattr(policy, "step_count", 0),
            "loss_history": getattr(policy, "loss_history", []),
            "weights": weights_dict,
            "learning_rate": getattr(policy, "learning_rate", 0.001)
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Also update 'latest.json' symlink / copy for automatic restore
        latest_path = user_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"PolicyCheckpointManager: Saved policy checkpoint for '{user_id}' to {save_path}")
        return save_path

    def load_checkpoint(
        self,
        policy: Any,
        user_id: str = "default_user",
        checkpoint_path: Optional[str] = None
    ) -> bool:
        """
        Restore weights and optimizer state into the active policy instance.
        """
        if checkpoint_path is not None:
            target_path = Path(checkpoint_path)
        else:
            target_path = self._get_user_dir(user_id) / "latest.json"

        if not target_path.exists():
            logger.warning(f"PolicyCheckpointManager: Checkpoint {target_path} does not exist.")
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            weights = data.get("weights", {})
            if hasattr(policy, "W") and "W" in weights:
                policy.W = np.array(weights["W"], dtype=np.float32)
                policy.b = np.array(weights["b"], dtype=np.float32)
            elif hasattr(policy, "load_state_dict"):
                import torch
                state_dict = {k: torch.tensor(v) for k, v in weights.items()}
                policy.load_state_dict(state_dict)

            if "step_count" in data:
                policy.step_count = data["step_count"]
            if "loss_history" in data:
                policy.loss_history = data["loss_history"]

            logger.info(f"PolicyCheckpointManager: Loaded policy checkpoint from {target_path} (step={data.get('step_count', 0)})")
            return True
        except Exception as e:
            logger.error(f"PolicyCheckpointManager: Error loading checkpoint from {target_path}: {e}")
            return False

    def reset_to_baseline(self, policy: Any) -> None:
        """Reset policy residual weights and buffer back to zero baseline."""
        if hasattr(policy, "W") and hasattr(policy, "b"):
            policy.W.fill(0.0)
            policy.b.fill(0.0)
            policy.step_count = 0
            policy.loss_history.clear()
            if hasattr(policy, "replay_buffer"):
                policy.replay_buffer.clear()
        logger.info("PolicyCheckpointManager: Reset residual policy weights to baseline zero.")

    def list_checkpoints(self, user_id: str = "default_user") -> List[Path]:
        """List all checkpoint files available for user profile."""
        user_dir = self._get_user_dir(user_id)
        return sorted(list(user_dir.glob("*.json")))
