from typing import Iterable, Dict, List

import numpy as np

from baal import ActiveLearningDataset


class StoppingCriterion:
    def __init__(self, active_dataset: ActiveLearningDataset):
        self._active_ds = active_dataset

    def should_stop(self, metrics: Dict[str, float], uncertainty: Iterable[float]) -> bool:
        raise NotImplementedError


class LabellingBudgetStoppingCriterion(StoppingCriterion):
    """Stops when the labelling budget is exhausted."""

    def __init__(self, active_dataset: ActiveLearningDataset, labelling_budget: int):
        super().__init__(active_dataset)
        self._start_length = len(active_dataset)
        self.labelling_budget = labelling_budget

    def should_stop(self, metrics: Dict[str, float], uncertainty: Iterable[float]) -> bool:
        return (len(self._active_ds) - self._start_length) >= self.labelling_budget


class LowAverageUncertaintyStoppingCriterion(StoppingCriterion):
    """Stops when the average uncertainty is on average below a threshold."""

    def __init__(self, active_dataset: ActiveLearningDataset, avg_uncertainty_thresh: float):
        super().__init__(active_dataset)
        self.avg_uncertainty_thresh = avg_uncertainty_thresh

    def should_stop(self, metrics: Dict[str, float], uncertainty: Iterable[float]) -> bool:
        arr = np.array(uncertainty)
        return bool(np.mean(arr) < self.avg_uncertainty_thresh)


class EarlyStoppingCriterion(StoppingCriterion):
    """Early stopping on a particular metrics.

    Notes:
    We don't have any mandatory dependency with an early stopping implementation.
    So we have our own.
    """

    def __init__(
        self,
        active_dataset: ActiveLearningDataset,
        metric_name: str,
        patience: int = 10,
        epsilon: float = 1e-4,
    ):
        super().__init__(active_dataset)
        self.metric_name = metric_name
        self.patience = patience
        self.epsilon = epsilon
        self._acc: List[float] = []

    def should_stop(self, metrics: Dict[str, float], uncertainty: Iterable[float]) -> bool:
        self._acc.append(metrics[self.metric_name])
        near_threshold = np.isclose(np.array(self._acc), self._acc[-1], atol=self.epsilon)
        return len(near_threshold) >= self.patience and bool(
            near_threshold[-(self.patience + 1) :].all()
        )
