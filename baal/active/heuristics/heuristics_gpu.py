from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from baal import ModelWrapper

available_reductions = {
    "max": lambda x: torch.max(
        x.view([x.shape[0], -1]),
        -1,
    ),
    "min": lambda x: torch.min(
        x.view([x.shape[0], -1]),
        -1,
    ),
    "mean": lambda x: torch.mean(
        x.view([x.shape[0], -1]),
        -1,
    ),
    "sum": lambda x: torch.sum(
        x.view([x.shape[0], -1]),
        -1,
    ),
    "none": lambda x: x,
}


def _shuffle_subset(data: torch.Tensor, shuffle_prop: float) -> torch.Tensor:
    to_shuffle = np.nonzero(np.random.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[np.random.permutation(to_shuffle), ...]
    return data


def requireprobs(fn):
    """Will convert logits to probs if needed"""

    def wrapper(self, probabilities):
        # Expected shape : [n_sample, n_classes, ..., n_iterations]
        bounded = torch.min(probabilities) < 0 or torch.max(probabilities) > 1.0
        if bounded or not probabilities.sum(1).allclose(1):
            probabilities = F.softmax(probabilities, 1)
        return fn(self, probabilities)

    return wrapper


class AbstractGPUHeuristic(ModelWrapper):
    """Abstract class that defines a Heuristic.

    Args:
        shuffle_prop (float): shuffle proportion.
        threshold (Optional[float]): threshold the probabilities.
        reverse (bool): True if the most uncertain sample has the highest value.
        reduction (Union[str, Callable]): Reduction used after computing the score.
    """

    def __init__(
        self,
        model: ModelWrapper,
        criterion,
        shuffle_prop=0.0,
        threshold=None,
        reverse=False,
        reduction="none",
    ):
        super().__init__(model, criterion)
        self.shuffle_prop = shuffle_prop
        self.threshold = threshold
        self.reversed = reverse
        assert reduction in available_reductions or callable(reduction)
        self.reduction = reduction if callable(reduction) else available_reductions[reduction]

    def compute_score(self, predictions):
        """
        Compute the score according to the heuristic.
        Args:
            predictions (ndarray): Array of predictions

        Returns:
            Array of scores.
        """
        raise NotImplementedError

    def get_uncertainties(self, predictions):
        """Get the uncertainties"""
        scores = self.compute_score(predictions)
        scores = self.reduction(scores)
        scores[~torch.isfinite(scores)] = 0.0 if self.reversed else 10000
        return scores

    def predict_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int,
        use_cuda: bool,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        return (
            super()
            .predict_on_dataset(
                dataset, batch_size, iterations, use_cuda, workers, collate_fn, half, verbose
            )
            .reshape([-1])
        )

    def predict_on_batch(self, data, iterations=1, use_cuda=False):
        """Rank the predictions according to their uncertainties."""
        return self.get_uncertainties(self.model.predict_on_batch(data, iterations, cuda=use_cuda))


class BALDGPUWrapper(AbstractGPUHeuristic):
    """Sort by the highest acquisition function value.
    References:
        https://arxiv.org/abs/1703.02910
    """

    def __init__(
        self, model: ModelWrapper, criterion, shuffle_prop=0.0, threshold=None, reduction="none"
    ):
        super().__init__(
            model,
            criterion=criterion,
            shuffle_prop=shuffle_prop,
            threshold=threshold,
            reverse=True,
            reduction=reduction,
        )

    @requireprobs
    def compute_score(self, predictions):
        assert predictions.ndimension() >= 3
        # [n_sample, n_class, ..., n_iterations]
        expected_entropy = -torch.mean(
            torch.sum(predictions * torch.log(predictions + 1e-5), 1), dim=-1
        )  # [batch size, ...]
        expected_p = torch.mean(predictions, dim=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = -torch.sum(
            expected_p * torch.log(expected_p + 1e-5), dim=1
        )  # [batch size, ...]
        bald_acq = entropy_expected_p - expected_entropy
        return bald_acq
