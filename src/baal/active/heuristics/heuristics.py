import types
import warnings
from functools import wraps as _wraps
from typing import List
from collections.abc import Sequence

import numpy as np
import scipy.stats
from scipy.special import xlogy
from torch import Tensor

from baal.utils.array_utils import to_prob

available_reductions = {
    'max': lambda x: np.max(x, axis=tuple(range(1, x.ndim))),
    'min': lambda x: np.min(x, axis=tuple(range(1, x.ndim))),
    'mean': lambda x: np.mean(x, axis=tuple(range(1, x.ndim))),
    'sum': lambda x: np.sum(x, axis=tuple(range(1, x.ndim))),
    'none': lambda x: x,
}


def _shuffle_subset(data: np.ndarray, shuffle_prop: float) -> np.ndarray:
    to_shuffle = np.nonzero(np.random.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[np.random.permutation(to_shuffle), ...]
    return data


def singlepass(fn):
    """
    Will take the mean of the iterations if needed.

    Args:
        fn (Callable): Heuristic function.

    Returns:
        fn : Array -> Array

    """

    @_wraps(fn)
    def wrapper(self, probabilities):
        if probabilities.ndim >= 3:
            # Expected shape : [n_sample, n_classes, ..., n_iterations]
            probabilities = probabilities.mean(-1)
        return fn(self, probabilities)

    return wrapper


def requireprobs(fn):
    """
    Will convert logits to probs if needed.

    Args:
        fn (Fn): Function that takes logits as input to wraps.

    Returns:
        Wrapper function
    """

    @_wraps(fn)
    def wrapper(self, probabilities):
        # Expected shape : [n_sample, n_classes, ..., n_iterations]
        probabilities = to_prob(probabilities)
        return fn(self, probabilities)

    return wrapper


class AbstractHeuristic:
    """
    Abstract class that defines a Heuristic.

    Args:
        shuffle_prop (float): shuffle proportion.
        reverse (bool): True if the most uncertain sample has the highest value.
        reduction (Union[str, Callable]): Reduction used after computing the score.
    """

    def __init__(self, shuffle_prop=0.0, reverse=False, reduction='none'):
        self.shuffle_prop = shuffle_prop
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

    def get_uncertainties_generator(self, predictions):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (Iterable): Generator of predictions

        Raises:
            ValueError if the generator is empty.

        Returns:
            Array of scores.
        """
        acc = []
        for pred in predictions:
            acc.append(self.get_uncertainties(pred))
        if len(acc) == 0:
            raise ValueError('No prediction! Cannot order the values!')
        return np.concatenate(acc)

    def get_uncertainties(self, predictions):
        """
        Get the uncertainties.

        Args:
            predictions (ndarray): Array of predictions

        Returns:
            Array of uncertainties

        """
        if isinstance(predictions, Tensor):
            predictions = predictions.numpy()
        scores = self.compute_score(predictions)
        scores = self.reduction(scores)
        if not np.all(np.isfinite(scores)):
            fixed = 0.0 if self.reversed else 10000
            warnings.warn(f"Invalid value in the score, will be put to {fixed}", UserWarning)
            scores[~np.isfinite(scores)] = fixed
        return scores

    def reorder_indices(self, scores):
        """
        Order indices given their uncertainty score.

        Args:
            scores (ndarray/ List[ndarray]): Array of uncertainties or
                list of arrays.

        Returns:
            ordered index according to the uncertainty (highest to lowes).
        """
        if isinstance(scores, Sequence):
            scores = np.concatenate(scores)
        assert scores.ndim <= 2
        ranks = np.argsort(scores, -1)
        if self.reversed:
            ranks = ranks[::-1]
        ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks

    def get_ranks(self, predictions):
        """
        Rank the predictions according to their uncertainties.

        Args:
            predictions (ndarray): [batch_size, C, ..., Iterations]

        Returns:
            Ranked index according to the uncertainty (highest to lowes).

        """
        if isinstance(predictions, types.GeneratorType):
            scores = self.get_uncertainties_generator(predictions)
        else:
            scores = self.get_uncertainties(predictions)

        return self.reorder_indices(scores)

    def __call__(self, predictions):
        """Rank the predictions according to their uncertainties."""
        return self.get_ranks(predictions)


class BALD(AbstractHeuristic):
    """
    Sort by the highest acquisition function value.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none`).

    References:
        https://arxiv.org/abs/1703.02910
    """

    def __init__(self, shuffle_prop=0.0, reduction='none'):
        super().__init__(
            shuffle_prop=shuffle_prop, reverse=True, reduction=reduction
        )

    @requireprobs
    def compute_score(self, predictions):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions

        Returns:
            Array of scores.
        """
        assert predictions.ndim >= 3
        # [n_sample, n_class, ..., n_iterations]

        expected_entropy = - np.mean(np.sum(xlogy(predictions, predictions), axis=1),
                                     axis=-1)  # [batch size, ...]
        expected_p = np.mean(predictions, axis=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = - np.sum(xlogy(expected_p, expected_p),
                                      axis=1)  # [batch size, ...]
        bald_acq = entropy_expected_p - expected_entropy
        return bald_acq


class BatchBALD(BALD):
    """
    Implementation of BatchBALD.

    Args:
        num_samples (int): Number of samples to select. (min 2*the amount of samples you want)
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none').

    References:
        https://arxiv.org/abs/1906.08158

    Notes:
        K = iterations, C=classes
        Not tested on 4+ dims.
        """

    def __init__(self, num_samples, shuffle_prop=0.0, reduction='none'):
        self.epsilon = 1e-5
        self.num_samples = num_samples
        super().__init__(shuffle_prop=shuffle_prop,
                         reduction=reduction)

    def _conditional_entropy(self, probs):
        K = probs.shape[-1]
        return np.sum(-xlogy(probs, probs), axis=(1, -1)) / K

    def _joint_entropy(self, predictions, selected):
        K = predictions.shape[-1]
        M = selected.shape[0]

        exp_y = np.array(
            [np.matmul(selected, predictions[i].T) for i in range(predictions.shape[0])]) / K
        mean_entropy = selected.mean(-1, keepdims=True)[None]

        step = 256
        for idx in range(0, exp_y.shape[0], step):
            b_preds = exp_y[idx: (idx + step)]
            yield np.sum(-xlogy(b_preds, b_preds) / mean_entropy, axis=(1, -1)) / M

    @requireprobs
    def compute_score(self, predictions):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions

        Returns:
            Array of scores.
        """
        MAX_SELECTED = 16000
        MIN_SPREAD = 0.1
        COUNT = 0
        # Get conditional_entropies_B
        conditional_entropies_B = self._conditional_entropy(predictions)
        bald_out = super().compute_score(predictions)
        history = [self.reduction(bald_out).argmax()]
        for step in range(2 * self.num_samples):
            # Select M, iterations example from history, take entropy
            selected = predictions[np.random.permutation(history)[:MAX_SELECTED]]
            selected = self.reduction(scipy.stats.entropy(np.swapaxes(selected, 0, 1)))
            # Compute join entropy
            joint_entropy = list(self._joint_entropy(predictions, selected))
            joint_entropy = np.concatenate(joint_entropy)

            partial_multi_bald_b = joint_entropy - conditional_entropies_B
            partial_multi_bald_b = self.reduction(partial_multi_bald_b)
            partial_multi_bald_b[..., np.array(history)] = -1
            # Add best to history
            partial_multi_bald_b = partial_multi_bald_b.squeeze()
            assert partial_multi_bald_b.ndim == 1
            winner_index = partial_multi_bald_b.argmax()
            history.append(winner_index)
            if partial_multi_bald_b.max() < MIN_SPREAD:
                COUNT += 1
                if COUNT > 50 or len(history) >= predictions.shape[0]:
                    break

        return np.array(history)

    def reorder_indices(self):
        """This function is not supported by BatchBald.

        Raises:
            All the time.
        """
        raise Exception("BatchBald needs to have the whole pool at once,"
                        "to be able to have relevant informationa chunk"
                        " processing is not supported by BatchBald")

    def get_ranks(self, predictions):
        """
        Rank the predictions according to their uncertainties.

        Args:
            predictions (ndarray): [batch_size, C, ..., Iterations]

        Returns:
            Ranked index according to the uncertainty (highest to lowest).

        Raises:
            ValueError if predictions is a generator.
        """
        if isinstance(predictions, types.GeneratorType):
            raise ValueError("BatchBALD doesn't support generators.")

        ranks = self.get_uncertainties(predictions)

        assert ranks.ndim == 1
        ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks


class Variance(AbstractHeuristic):
    """
    Sort by the highest variance.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results (default: `mean`).
    """

    def __init__(self, shuffle_prop=0.0, reduction='mean'):
        _help = "Need to reduce the output from [n_sample, n_class] to [n_sample]"
        assert reduction != 'none', _help
        super().__init__(
            shuffle_prop=shuffle_prop, reverse=True, reduction=reduction
        )

    def compute_score(self, predictions):
        assert predictions.ndim >= 3
        return np.var(predictions, -1)


class Entropy(AbstractHeuristic):
    """
    Sort by the highest entropy.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results (default: `none`).
    """

    def __init__(self, shuffle_prop=0.0, reduction='none'):
        super().__init__(
            shuffle_prop=shuffle_prop, reverse=True, reduction=reduction
        )

    @singlepass
    @requireprobs
    def compute_score(self, predictions):
        return scipy.stats.entropy(np.swapaxes(predictions, 0, 1))


class Margin(AbstractHeuristic):
    """
    Sort by the lowest margin, i.e. the difference between the most confident class and
    the second most confident class.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: `none`).
    """

    def __init__(self, shuffle_prop=0.0, reduction='none'):
        super().__init__(
            shuffle_prop=shuffle_prop, reverse=False, reduction=reduction
        )

    @singlepass
    @requireprobs
    def compute_score(self, predictions):
        sort_arr = np.sort(predictions, axis=1)
        return sort_arr[:, -1] - sort_arr[:, -2]


class Certainty(AbstractHeuristic):
    """
    Sort by the lowest certainty.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias.
        reduction (Union[str, callable]): function that aggregates the results.
    """

    def __init__(self, shuffle_prop=0.0, reduction='none'):
        super().__init__(
            shuffle_prop=shuffle_prop, reverse=False, reduction=reduction
        )

    @singlepass
    def compute_score(self, predictions):
        return np.max(predictions, axis=1)


class Precomputed(AbstractHeuristic):
    """Precomputed heuristics.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias.
        reverse (Bool): Sort from lowest to highest if False.
    """

    def __init__(self, shuffle_prop=0.0, reverse=False):
        super().__init__(shuffle_prop, reverse=reverse)

    def compute_score(self, predictions):
        return predictions


class Random(Precomputed):
    """Random heuristic.

    Args:
        shuffle_prop (float): UNUSED
        reduction (Union[str, callable]): UNUSED.
    """

    def __init__(self, shuffle_prop=0.0, reduction='none'):
        super().__init__(1.0, False)

    def reorder_indices(self, predictions):
        """
        Order indices randomly.

        Args:
            predictions (ndarray): predictions for samples

        Returns:
            ranked indices (randomly)
        """
        if isinstance(predictions, Sequence):
            predictions = np.concatenate(predictions)
        ranks = np.arange(predictions.shape[0])
        ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks

    def get_ranks(self, predictions):
        if isinstance(predictions, types.GeneratorType):
            predictions = np.array([1 for _ in predictions])
        return self.reorder_indices(predictions)


class CombineHeuristics(AbstractHeuristic):
    """Combine heuristics for multi-output models.
    heuristics would be applied on output predictions in the assigned order.
    For each heuristic the necessary `reduction`, `reversed`
    parameters should be defined.

    NOTE: heuristics could be combined together only if they use the same
    value for `reversed` parameter.

    NOTE: `shuffle_prop` should only be defined as direct input of
    `CombineHeuristics`, otherwise there will be no effect.

    NOTE: `reduction` is defined for each of the input heuristics and as a direct
    input to `CombineHeuristics`. For each heuristic, `reduction` should be defined
    if the relevant model output to that heuristic has more than 3-dimenstions.
    In `CombineHeuristics`, the `reduction` is used to aggregate the final result of
    heuristics.

    Args:
        heuristics (list[AbstractHeuristic]): list of heuristic instances
        weights (list[float]): the assigned weights to the result of each heuristic
            before calculation of ranks
        reduction (Union[str, callable]): function that aggregates the results of the heuristics
            (default: weighted average which could be used as (reduction='mean`)
       shuffle_prop (float): shuffle proportion.

    """
    def __init__(self, heuristics: List, weights: List, reduction='mean', shuffle_prop=0.0):
        super(CombineHeuristics, self).__init__(reduction=reduction, shuffle_prop=shuffle_prop)
        self.composed_heuristic = heuristics
        self.weights = weights
        reversed = [bool(heuristic.reversed) for heuristic in self.composed_heuristic]

        if all(item is False for item in reversed):
            self.reversed = False
        elif all(item is True for item in reversed):
            self.reversed = True
        else:
            raise Exception("heuristics should have the same value for `revesed` parameter")

    def get_uncertainties(self, predictions):
        """
        Computes the score for each part of predictions according to the assigned heuristic.

        NOTE: predictions is a list of each model outputs. For example for a object detection model,
        the predictions should be as:
            [confidence_predictions: nd.array(), boundingbox_predictions: nd.array()]

        Args:
            predictions (list[ndarray]): list of predictions arrays

        Returns:
            Array of uncertainties

        """

        results = []
        for ind, prediction in enumerate(predictions):
            if isinstance(predictions[0], types.GeneratorType):
                results.append(self.composed_heuristic[ind].get_uncertainties_generator(prediction))
            else:
                results.append(self.composed_heuristic[ind].get_uncertainties(prediction))
        return results

    def reorder_indices(self, scores_list):
        """
        Order the indices based on the given scores.

        Args:
            scores_list (list(ndarray)/list(list(ndarray)):

        Returns:
            ordered index according to the uncertainty (highest to lowes).

        """
        if isinstance(scores_list[0], Sequence):
            scores_list = list(zip(*scores_list))
            scores_list = [np.concatenate(item) for item in scores_list]

        # normalizing weights
        w = np.array(self.weights).sum()
        self.weights = [weight / w for weight in self.weights]

        # num_heuristics X batch_size
        scores_array = np.vstack([weight * scores
                                  for weight, scores in zip(self.weights, scores_list)])

        # batch_size X num_heuristic
        final_scores = self.reduction(np.swapaxes(scores_array, 0, -1))

        assert final_scores.ndim == 1
        ranks = np.argsort(final_scores)

        if self.reversed:
            ranks = ranks[::-1]
        ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks

    def get_ranks(self, predictions):
        """
        Rank the predictions according to the weighted vote of each heuristic.

        Args:
            predictions (list[ndarray]):
                list[[batch_size, C, ..., Iterations], [batch_size, C, ..., Iterations], ...]

        Returns:
            Ranked index according to the uncertainty (highest to lowest).

        """

        scores_list = self.get_uncertainties(predictions)

        return self.reorder_indices(scores_list)
