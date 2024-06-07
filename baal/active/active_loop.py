import os
import pickle
import types
import warnings
from typing import Callable

import numpy as np
import structlog
import torch.utils.data as torchdata

from . import heuristics
from .dataset import ActiveLearningDataset

log = structlog.get_logger("baal")
pjoin = os.path.join


class ActiveLearningLoop:
    """Object that perform the active learning iteration.

    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        get_probabilities (Function): Dataset -> **kwargs ->
                                        ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        query_size (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        uncertainty_folder (Optional[str]): If provided, will store uncertainties on disk.
        ndata_to_label (int): DEPRECATED, please use `query_size`.
        **kwargs: Parameters forwarded to `get_probabilities`.
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        query_size: int = 1,
        max_sample=-1,
        uncertainty_folder=None,
        ndata_to_label=None,
        **kwargs,
    ) -> None:
        if ndata_to_label is not None:
            warnings.warn(
                "`ndata_to_label` is deprecated, please use `query_size`.", DeprecationWarning
            )
            query_size = ndata_to_label
        self.query_size = query_size
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.dataset = dataset
        self.max_sample = max_sample
        self.uncertainty_folder = uncertainty_folder
        self.kwargs = kwargs

    def step(self, pool=None) -> bool:
        """
        Perform an active learning step.

        Args:
            pool (iterable): Optional dataset pool indices.
                             If not set, will use pool from the active set.

        Returns:
            boolean, Flag indicating if we continue training.

        """
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
            if isinstance(self.heuristic, heuristics.Random):
                probs = np.random.uniform(low=0, high=1, size=(len(pool), 1))
                target_probs = None
            else:
                probs = self.get_probabilities(pool, **self.kwargs)
                if isinstance(self.heuristic, heuristics.EPIG):
                    target_probs = self.get_probabilities(self.dataset, **self.kwargs)
                else:
                    target_probs = None
            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                to_label, uncertainty = self.heuristic.get_ranks(probs, target_probs)
                log.info(
                    "Uncertainty",
                    mean=uncertainty.mean(),
                    std=uncertainty.std(),
                    median=np.median(uncertainty),
                )
                if indices is not None:
                    to_label = indices[np.array(to_label)]
                if self.uncertainty_folder is not None:
                    # We save uncertainty in a file.
                    uncertainty_name = (
                        f"uncertainty_pool={len(pool)}" f"_labelled={len(self.dataset)}.pkl"
                    )
                    pickle.dump(
                        {
                            "indices": indices,
                            "uncertainty": uncertainty,
                            "dataset": self.dataset.state_dict(),
                        },
                        open(pjoin(self.uncertainty_folder, uncertainty_name), "wb"),
                    )
                if len(to_label) > 0:
                    self.dataset.label(to_label[: self.query_size])
                    return True
        return False
