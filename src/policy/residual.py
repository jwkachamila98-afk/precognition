"""Residual adaptation loop interfaces."""

from abc import ABC, abstractmethod
import numpy as np
from src.policy.discrepancy import DiscrepancyMetric


class ResidualAdaptationABC(ABC):
    """Abstract Base Class for adapting nominal actions using discrepancy signals."""

    @abstractmethod
    def adapt(self, nominal_action: np.ndarray, discrepancy: DiscrepancyMetric) -> np.ndarray:
        """
        Adjust nominal action based on discrepancy metrics.

        Args:
            nominal_action: Baseline action from trajectory planner or policy.
            discrepancy: DiscrepancyMetric error measurements.

        Returns:
            Adapted action array.
        """
        pass
