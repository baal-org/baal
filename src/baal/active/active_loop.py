import types
from typing import Callable

import numpy as np
import torch.utils.data as torchdata

from . import heuristics
from .dataset import ActiveLearningDataset


class ActiveLearningLoop:
    """Object that perform the active learning iteration.

    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        get_probabilities (Function): Dataset -> **kwargs ->
                                        ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        ndata_to_label (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        **kwargs: Parameters forwarded to `get_probabilities`.
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        ndata_to_label: int = 1,
        max_sample=-1,
        **kwargs,
    ) -> None:
        self.ndata_to_label = ndata_to_label
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.dataset = dataset
        self.max_sample = max_sample
        self.kwargs = kwargs

    def step(self, pool=None) -> bool:
        """
        Perform an active learning step.

        Args:
            pool (iterable): dataset pool indices.

        Returns:
            boolean, Flag indicating if we continue training.

        """
        # High to low
        if pool is None:
            pool = self.dataset.pool
            if len(pool) > 0:
                # Limit number of samples
                if self.max_sample != -1 and self.max_sample < len(pool):
                    indices = np.random.choice(len(pool), self.max_sample, replace=False)
                    pool = torchdata.Subset(pool, indices)
                else:
                    indices = np.arange(len(pool))
        else:
            indices = None

        if len(pool) > 0:
            probs = self.get_probabilities(pool, **self.kwargs)
            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                to_label = self.heuristic(probs)
                if indices is not None:
                    to_label = indices[np.array(to_label)]
                if len(to_label) > 0:
                    self.dataset.label(to_label[: self.ndata_to_label])
                    return True
        return False
