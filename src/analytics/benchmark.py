"""Multi-Trial Co-Adaptation Analytics and Benchmarking Suite."""

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.policy.discrepancy import EpisodeDiscrepancyReport

logger = logging.getLogger(__name__)


@dataclass
class TrialMetrics:
    """Quantitative performance measurements for a single manipulation trial."""
    trial_index: int
    intent: str
    mean_pose_error: float            # Cumulative trajectory discrepancy D_traj (meters)
    max_pose_error: float             # Peak tracking divergence (meters)
    smoothness_variance: float        # Motion jerk variance
    contact_misalignment: float       # Final contact error (meters)
    episode_reward: float             # Episode reward R_episode in [-1.0, 1.0]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "intent": self.intent,
            "mean_pose_error": float(self.mean_pose_error),
            "max_pose_error": float(self.max_pose_error),
            "smoothness_variance": float(self.smoothness_variance),
            "contact_misalignment": float(self.contact_misalignment),
            "episode_reward": float(self.episode_reward),
            "timestamp": float(self.timestamp)
        }


class CoAdaptationBenchmark:
    """
    Evaluates multi-trial human-robot co-adaptation progress over sequential trials 1...N.
    Tracks discrepancy convergence, reward trajectories, kinematic jerk, and generates
    terminal ASCII learning curves and exportable analytics.
    """

    def __init__(self, log_dir: str = "logs/benchmarks") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trials: List[TrialMetrics] = []

    @property
    def total_trials(self) -> int:
        return len(self.trials)

    def record_trial(
        self,
        report: EpisodeDiscrepancyReport,
        intent: str = ""
    ) -> TrialMetrics:
        """
        Ingest an episode report and append quantitative trial metrics.
        """
        trial_idx = len(self.trials) + 1
        metric = TrialMetrics(
            trial_index=trial_idx,
            intent=intent,
            mean_pose_error=report.mean_pose_error,
            max_pose_error=report.max_pose_error,
            smoothness_variance=report.smoothness_variance,
            contact_misalignment=report.contact_misalignment,
            episode_reward=report.episode_reward
        )
        self.trials.append(metric)
        logger.info(
            f"CoAdaptationBenchmark: Logged Trial {trial_idx} | "
            f"Error={metric.mean_pose_error*1000:.1f}mm | Reward={metric.episode_reward:+.3f}"
        )
        return metric

    def compute_error_reduction_pct(self) -> float:
        """Compute percentage error reduction from Trial 1 to latest Trial N."""
        if len(self.trials) < 2:
            return 0.0
        initial_err = self.trials[0].mean_pose_error
        latest_err = self.trials[-1].mean_pose_error
        if initial_err <= 1e-6:
            return 0.0
        return float(((initial_err - latest_err) / initial_err) * 100.0)

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate summary metrics across all recorded trials."""
        if not self.trials:
            return {
                "total_trials": 0,
                "error_reduction_pct": 0.0,
                "mean_reward": 0.0,
                "latest_error_mm": 0.0
            }

        errors = [t.mean_pose_error for t in self.trials]
        rewards = [t.episode_reward for t in self.trials]

        return {
            "total_trials": len(self.trials),
            "error_reduction_pct": self.compute_error_reduction_pct(),
            "mean_reward": float(np.mean(rewards)),
            "latest_reward": float(rewards[-1]),
            "initial_error_mm": float(errors[0] * 1000.0),
            "latest_error_mm": float(errors[-1] * 1000.0),
            "min_error_mm": float(np.min(errors) * 1000.0)
        }

    def format_ascii_trend_graph(self, max_width: int = 40) -> str:
        """
        Generate terminal ASCII learning curve demonstrating trajectory discrepancy reduction.
        """
        if not self.trials:
            return "No trials logged yet."

        lines = [
            "+----------------------------------------------------------------+",
            "|      CO-ADAPTATION MULTI-TRIAL LEARNING CONVERGENCE (D_traj)   |",
            "+-------+-------------+----------+-------------------------------+"
        ]
        lines.append("| Trial | Error (mm)  | Reward   | Relative Discrepancy Bar      |")
        lines.append("+-------+-------------+----------+-------------------------------+")

        max_err = max(t.mean_pose_error for t in self.trials) if self.trials else 1.0
        max_err = max(max_err, 0.01)

        for t in self.trials:
            err_mm = t.mean_pose_error * 1000.0
            bar_len = int((t.mean_pose_error / max_err) * (max_width // 2))
            bar_str = "█" * max(1, bar_len)
            lines.append(f"| #{t.trial_index:<4} | {err_mm:8.2f} mm | {t.episode_reward:+6.2f}   | {bar_str:<29} |")

        lines.append("+-------+-------------+----------+-------------------------------+")
        reduction = self.compute_error_reduction_pct()
        lines.append(f"| OVERALL ERROR REDUCTION: {reduction:+6.1f}% | TRIALS RECORDED: {len(self.trials):<10} |")
        lines.append("+----------------------------------------------------------------+")

        return "\n".join(lines)

    def export_summary_json(self, filepath: Optional[str] = None) -> Path:
        """Export benchmark summary and full trial records to JSON."""
        target = Path(filepath) if filepath else self.log_dir / f"benchmark_{int(time.time())}.json"
        data = {
            "summary": self.get_summary(),
            "trials": [t.to_dict() for t in self.trials]
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported benchmark analytics to: {target}")
        return target

    def export_csv(self, filepath: Optional[str] = None) -> Path:
        """Export trial records to CSV format."""
        target = Path(filepath) if filepath else self.log_dir / f"benchmark_{int(time.time())}.csv"
        if not self.trials:
            return target

        keys = list(self.trials[0].to_dict().keys())
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for t in self.trials:
                writer.writerow(t.to_dict())
        logger.info(f"Exported benchmark CSV to: {target}")
        return target
