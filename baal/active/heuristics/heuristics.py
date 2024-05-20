import types
import warnings
from collections.abc import Sequence
from functools import wraps as _wraps
from itertools import zip_longest
from typing import List

import numpy as np
import scipy.stats
import torch
from scipy.special import xlogy
from torch import Tensor

from baal.utils.array_utils import to_prob

DEPRECATED = "DEPRECATED"
SHUFFLE_PROP_DEPRECATION_NOTICE = """
`shuffle_prop` is deprecated and will be removed in Baal 1.9.0.
For better batch uncertainty estimation, use `baal.active.heuristics.stochastics.PowerSampling`.
See `https://baal.readthedocs.io/en/latest/user_guide/heuristics/` for more details.
"""

available_reductions = {
    "max": lambda x: np.max(x, axis=tuple(range(1, x.ndim))),
    "min": lambda x: np.min(x, axis=tuple(range(1, x.ndim))),
    "mean": lambda x: np.mean(x, axis=tuple(range(1, x.ndim))),
    "sum": lambda x: np.sum(x, axis=tuple(range(1, x.ndim))),
    "none": lambda x: x,
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
    def wrapper(self, probabilities, training_predictions=None):
        if probabilities.ndim >= 3:
            # Expected shape : [n_sample, n_classes, ..., n_iterations]
            probabilities = probabilities.mean(-1)
        if training_predictions is not None and training_predictions.ndim >= 3:
            training_predictions = training_predictions.mean(-1)
        return fn(self, probabilities, training_predictions)

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
    def wrapper(self, probabilities, training_predictions=None):
        # Expected shape : [n_sample, n_classes, ..., n_iterations]
        probabilities = to_prob(probabilities)
        training_predictions = (
            to_prob(training_predictions) if training_predictions is not None else None
        )
        return fn(self, probabilities, training_predictions=training_predictions)

    return wrapper


def require_single_item(fn):
    """
    Will check that the input is a single item.
    Useful when heuristics do not work on multi-output.

    Args:
        fn (Fn): Function that takes logits as input to wraps.

    Returns:
        Wrapper function

    """

    @_wraps(fn)
    def wrapper(self, probabilities, training_predictions=None):
        # Expected single shape : [n_sample, n_classes, ..., n_iterations]
        if isinstance(probabilities, (list, tuple)):
            if len(probabilities) == 1:
                probabilities = probabilities[0]
            else:
                raise ValueError(
                    "This heuristic accepts a single array with shape "
                    "[n_sample, n_classes, ..., n_iterations]. If you want"
                    " to compute uncertainty on a multi-model outputs,"
                    " we suggest using baal.active.heuristics.CombineHeuristics"
                )

        return fn(self, probabilities, training_predictions)

    return wrapper


def gather_expand(data, dim, index):
    """
    Gather indices `index` from `data` after expanding along dimension `dim`.

    Args:
        data (tensor): A tensor of data.
        dim (int): dimension to expand along.
        index (tensor): tensor with the indices to gather.

    References:
        Code from https://github.com/BlackHC/BatchBALD/blob/master/src/torch_utils.py

    Returns:
        Tensor with the same shape as `index`.
    """
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)


class AbstractHeuristic:
    """
    Abstract class that defines a Heuristic.

    Args:
        shuffle_prop (float): shuffle proportion.
        reverse (bool): True if the most uncertain sample has the highest value.
        reduction (Union[str, Callable]): Reduction used after computing the score.
    """

    def __init__(self, shuffle_prop=DEPRECATED, reverse=False, reduction="none"):
        if shuffle_prop != DEPRECATED and shuffle_prop < 1.0:
            warnings.warn(SHUFFLE_PROP_DEPRECATION_NOTICE, DeprecationWarning)
        else:
            shuffle_prop = 0.0
        self.shuffle_prop = shuffle_prop
        self.reversed = reverse
        assert reduction in available_reductions or callable(reduction)
        self._reduction_name = reduction
        self.reduction = reduction if callable(reduction) else available_reductions[reduction]

    def compute_score(self, predictions, training_predictions=None):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions
            training_predictions (Optional[ndarray]): Array of predictions on train set.

        Returns:
            Array of scores.
        """
        raise NotImplementedError

    def get_uncertainties_generator(self, predictions, training_predictions=None):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (Iterable): Generator of predictions
            training_predictions (Optional[ndarray]): Generator of training predictions

        Raises:
            ValueError if the generator is empty.

        Returns:
            Array of scores.
        """
        acc = []
        for pred in predictions:
            acc.append(self.get_uncertainties(pred, training_predictions=training_predictions))
        if len(acc) == 0:
            raise ValueError("No prediction! Cannot order the values!")
        return np.concatenate(acc)

    def get_uncertainties(self, predictions, training_predictions=None):
        """
        Get the uncertainties.

        Args:
            predictions (ndarray): Array of predictions
            training_predictions (ndarray): Array of predictions on training set.

        Returns:
            Array of uncertainties

        """
        if isinstance(predictions, Tensor):
            predictions = predictions.numpy()
        scores = self.compute_score(predictions, training_predictions=training_predictions)
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

        Raises:
            ValueError if `scores` is not uni-dimensional.
        """
        if isinstance(scores, Sequence):
            scores = np.concatenate(scores)

        if scores.ndim > 1:
            raise ValueError(
                (
                    f"Can't order sequence with more than 1 dimension."
                    f"Currently {scores.ndim} dimensions."
                    f"Is the heuristic reduction method set: {self._reduction_name}"
                )
            )
        assert scores.ndim == 1  # We want the uncertainty value per sample.
        ranks = np.argsort(scores)
        if self.reversed:
            ranks = ranks[::-1]
        ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks

    def get_ranks(self, predictions, training_predictions=None):
        """
        Rank the predictions according to their uncertainties.

        Args:
            predictions (ndarray): [batch_size, C, ..., Iterations]
            training_predictions (Optional[ndarray]): [batch_size, C, ..., Iterations]

        Returns:
            Ranked index according to the uncertainty (highest to lowes).
            Scores for all predictions.

        """
        if isinstance(predictions, types.GeneratorType):
            scores = self.get_uncertainties_generator(
                predictions, training_predictions=training_predictions
            )
        else:
            scores = self.get_uncertainties(predictions, training_predictions=training_predictions)

        return self.reorder_indices(scores), scores

    def __call__(self, predictions, training_predictions=None):
        """Rank the predictions according to their uncertainties.

        Only return the scores and not the associated uncertainties.
        """
        return self.get_ranks(predictions, training_predictions)[0]


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

    def __init__(self, shuffle_prop=DEPRECATED, reduction="none"):
        super().__init__(shuffle_prop=shuffle_prop, reverse=True, reduction=reduction)

    @require_single_item
    @requireprobs
    def compute_score(self, predictions, training_predictions=None):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions
            training_predictions (Optional[ndarray]): [batch_size, C, ..., Iterations]

        Returns:
            Array of scores.
        """
        assert predictions.ndim >= 3
        # [n_sample, n_class, ..., n_iterations]

        expected_entropy = -np.mean(
            np.sum(xlogy(predictions, predictions), axis=1), axis=-1
        )  # [batch size, ...]
        expected_p = np.mean(predictions, axis=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = -np.sum(xlogy(expected_p, expected_p), axis=1)  # [batch size, ...]
        bald_acq = entropy_expected_p - expected_entropy
        return bald_acq


class BatchBALD(BALD):
    """
    Implementation of BatchBALD from https://github.com/BlackHC/BatchBALD

    Args:
        num_samples (int): Number of samples to select (also called query_size).
        num_draw (int): Number of draw to perform from the history.
                        From the paper `40000 // num_classes` is suggested.
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none').

    Notes:
        This implementation returns the scores
         which is not necessarily ordered in the same way as they were selected.

    References:
        https://arxiv.org/abs/1906.08158

    Notes:
        K = iterations, C=classes
        Not tested on 4+ dims.
    """

    def __init__(self, num_samples, num_draw=500, shuffle_prop=DEPRECATED, reduction="none"):
        self.epsilon = 1e-5
        self.num_samples = num_samples
        self.num_draw = num_draw
        super().__init__(shuffle_prop=shuffle_prop, reduction=reduction)

    def _draw_choices(self, probs, n_choices):
        """
        Draw `n_choices` sample from `probs`.

        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/torch_utils.py#L187

        Returns:
            choices: B... x `n_choices`

        """
        probs = probs.permute(0, 2, 1)
        probs_B_C = probs.reshape((-1, probs.shape[-1]))

        # samples: Ni... x draw_per_xx
        choices = torch.multinomial(probs_B_C, num_samples=n_choices, replacement=True)

        choices_b_M = choices.reshape(list(probs.shape[:-1]) + [n_choices])
        return choices_b_M.long()

    def _sample_from_history(self, probs, num_draw=1000):
        """
        Sample `num_draw` choices from `probs`

        Args:
            probs (Tensor[batch, classes, ..., iterations]): Tensor to be sampled from.
            num_draw (int): Number of draw.

        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/joint_entropy/sampling.py

        Returns:
            Tensor[num_draw, iterations]
        """
        probs = torch.from_numpy(probs).double()

        n_iterations = probs.shape[-1]

        # [batch, draw, iterations]
        choices = self._draw_choices(probs, num_draw)

        # [batch, iterations, iterations, draw]
        expanded_choices_N_K_K_S = choices[:, None, :, :]
        expanded_probs_N_K_K_C = probs.permute(0, 2, 1)[:, :, None, :]

        probs = gather_expand(expanded_probs_N_K_K_C, dim=-1, index=expanded_choices_N_K_K_S)
        # exp sum log seems necessary to avoid 0s?
        entropies = torch.exp(torch.sum(torch.log(probs), dim=0, keepdim=False))
        entropies = entropies.reshape((n_iterations, -1))

        samples_M_K = entropies.t()
        return samples_M_K.numpy()

    def _conditional_entropy(self, probs):
        K = probs.shape[-1]
        return np.sum(-xlogy(probs, probs), axis=(1, -1)) / K

    def _joint_entropy(self, predictions, selected):
        """
        Compute the joint entropy between `preditions` and `selected`
        Args:
            predictions (Tensor): First tensor with shape [B, C, Iterations]
            selected (Tensor): Second tensor with shape [M, Iterations].

        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/joint_entropy/sampling.py

        Notes:
            Only Classification is supported, not semantic segmentation or other.

        Returns:
            Generator yield B entropies.
        """
        K = predictions.shape[-1]
        C = predictions.shape[1]
        B = predictions.shape[0]
        M = selected.shape[0]
        predictions = predictions.swapaxes(1, 2)

        exp_y = np.matmul(selected, predictions) / K
        assert exp_y.shape == (B, M, C)
        mean_entropy = selected.mean(-1, keepdims=True)[None]
        assert mean_entropy.shape == (1, M, 1)

        step = 10_000
        for idx in range(0, exp_y.shape[0], step):
            b_preds = exp_y[idx : idx + step]
            yield np.sum(-xlogy(b_preds, b_preds) / mean_entropy, axis=(1, -1)) / M

    @require_single_item
    @requireprobs
    def compute_score(self, predictions, training_predictions=None):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions [batch_size, C, Iterations]
            training_predictions (Optional[ndarray]): [batch_size, C, ..., Iterations]

        Notes:
            Only Classification is supported, not semantic segmentation or other.

        Returns:
            Array of scores.
        """
        MIN_SPREAD = 0.1
        COUNT = 0
        # Get conditional_entropies_B
        conditional_entropies_B = self._conditional_entropy(predictions)
        bald_out = super().compute_score(predictions)
        # We start with the most uncertain sample according to BALD.
        bald_out = self.reduction(bald_out)
        history = bald_out.argsort()[-1:].tolist()
        uncertainties = np.zeros_like(bald_out)
        uncertainties[history[0]] = bald_out.max()
        for step in range(self.num_samples):
            # Draw `num_draw` example from history, take entropy
            # TODO use numpy/numba
            selected = self._sample_from_history(predictions[history], num_draw=self.num_draw)

            # Compute join entropy
            joint_entropy = list(self._joint_entropy(predictions, selected))
            joint_entropy = np.concatenate(joint_entropy)

            partial_multi_bald_b = joint_entropy - conditional_entropies_B
            partial_multi_bald_b = self.reduction(partial_multi_bald_b)
            partial_multi_bald_b[..., np.array(history)] = -1000
            # Add best to history
            partial_multi_bald_b = partial_multi_bald_b.squeeze()
            assert partial_multi_bald_b.ndim == 1
            winner_index = partial_multi_bald_b.argmax()
            history.append(winner_index)
            uncertainties[winner_index] = partial_multi_bald_b.max()

            if partial_multi_bald_b.max() < MIN_SPREAD:
                COUNT += 1
                if COUNT > 10 or len(history) >= self.num_samples:
                    break

        return uncertainties

    def get_ranks(self, predictions, training_predictions=None):
        """
        Rank the predictions according to their uncertainties.

        Args:
            predictions (ndarray): [batch_size, C, Iterations]
            training_predictions (Optional[ndarray]): [batch_size, C, Iterations]

        Returns:
            Ranked index according to the uncertainty (highest to lowest).

        Notes:
            Only Classification is supported, not semantic segmentation or other.

        Raises:
            ValueError if predictions is a generator.
        """
        if isinstance(predictions, types.GeneratorType):
            raise ValueError("BatchBALD doesn't support generators.")

        if predictions.ndim != 3:
            raise ValueError(
                "BatchBALD only works on classification"
                "Expected shape= [batch_size, C, Iterations]"
            )

        return super().get_ranks(predictions)


class Variance(AbstractHeuristic):
    """
    Sort by the highest variance.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results (default: `mean`).
    """

    def __init__(self, shuffle_prop=DEPRECATED, reduction="mean"):
        _help = "Need to reduce the output from [n_sample, n_class] to [n_sample]"
        assert reduction != "none", _help
        super().__init__(shuffle_prop=shuffle_prop, reverse=True, reduction=reduction)

    @require_single_item
    def compute_score(self, predictions, training_predictions=None):
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

    def __init__(self, shuffle_prop=DEPRECATED, reduction="none"):
        super().__init__(shuffle_prop=shuffle_prop, reverse=True, reduction=reduction)

    @require_single_item
    @singlepass
    @requireprobs
    def compute_score(self, predictions, training_predictions=None):
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

    def __init__(self, shuffle_prop=DEPRECATED, reduction="none"):
        super().__init__(shuffle_prop=shuffle_prop, reverse=False, reduction=reduction)

    @require_single_item
    @singlepass
    @requireprobs
    def compute_score(self, predictions, training_predictions=None):
        sort_arr = np.sort(predictions, axis=1)
        return sort_arr[:, -1] - sort_arr[:, -2]


class Certainty(AbstractHeuristic):
    """
    Sort by the lowest certainty.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias.
        reduction (Union[str, callable]): function that aggregates the results.
    """

    def __init__(self, shuffle_prop=DEPRECATED, reduction="none"):
        super().__init__(shuffle_prop=shuffle_prop, reverse=False, reduction=reduction)

    @require_single_item
    @singlepass
    def compute_score(self, predictions, training_predictions=None):
        return np.max(predictions, axis=1)


class Precomputed(AbstractHeuristic):
    """Precomputed heuristics.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias.
        reverse (Bool): Sort from lowest to highest if False.
    """

    def __init__(self, shuffle_prop=DEPRECATED, reverse=False):
        super().__init__(shuffle_prop, reverse=reverse)

    def compute_score(self, predictions, training_predictions=None):
        return predictions


class Random(Precomputed):
    """Random heuristic.

    Args:
        shuffle_prop (float): UNUSED
        reduction (Union[str, callable]): UNUSED.
        seed (Optional[int]): If provided, will seed the random generator.
    """

    def __init__(self, shuffle_prop=DEPRECATED, reduction="none", seed=None):
        super().__init__(1.0, False)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def compute_score(self, predictions, training_predictions=None):
        return self.rng.rand(predictions.shape[0])


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

    def __init__(self, heuristics: List, weights: List, reduction="mean", shuffle_prop=DEPRECATED):
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

    def get_uncertainties(self, predictions, training_predictions=None):
        """
        Computes the score for each part of predictions according to the assigned heuristic.

        NOTE: predictions is a list of each model outputs. For example for a object detection model,
        the predictions should be as:
            [confidence_predictions: nd.array(), boundingbox_predictions: nd.array()]

        Args:
            predictions (list[ndarray]): list of predictions arrays
            training_predictions (Optional[List[ndarray]): List of predictions on training dataset.

        Returns:
            Array of uncertainties

        """

        results = []
        for ind, (prediction, train_pred) in enumerate(
            zip_longest(predictions, training_predictions or [], fillvalue=None)
        ):
            if isinstance(predictions[0], types.GeneratorType):
                results.append(
                    self.composed_heuristic[ind].get_uncertainties_generator(
                        prediction, training_predictions=train_pred
                    )
                )
            else:
                results.append(
                    self.composed_heuristic[ind].get_uncertainties(
                        prediction, training_predictions=train_pred
                    )
                )
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
        scores_array = np.vstack(
            [weight * scores for weight, scores in zip(self.weights, scores_list)]
        )

        # batch_size X num_heuristic
        final_scores = self.reduction(np.swapaxes(scores_array, 0, -1))

        assert final_scores.ndim == 1
        ranks = np.argsort(final_scores)

        if self.reversed:
            ranks = ranks[::-1]
        ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks


class EPIG(AbstractHeuristic):
    """
    Implementation of Expected Predicted Information Gain (https://arxiv.org/abs/2304.08151)

    Args:
        shuffle_prop (float): DEPRECATED
        reverse (bool): UNUSED
        reduction (Union[str, callable]): function that aggregates the results.

    References:
        Code from https://github.com/fbickfordsmith/epig
    """

    def __init__(self, shuffle_prop=DEPRECATED, reverse=False, reduction="none"):
        super().__init__(shuffle_prop=shuffle_prop, reverse=True, reduction=reduction)

    def marginal_entropy_from_probs(self, probs):
        """
        Compute the marginal predictive entropy for each input, x_i:
            H[p(y|x_i)]  = H[E_{q(θ)}[p(y|x_i,θ)]]
                        ~= H[(1/K) Σ_{j=1}^K p(y|x_i,θ_j)]
        where θ_j ~ q(θ) is a parameter sample and p(y|x_i,θ_j) is the parameter-conditional
        predictive distribution for x_i and θ_j.

        Args:
            probs (Tensor[float], [N, C, K]): p(y|x_i,θ_j) for i in [1, N] and j in [1, K].

        Returns:
            scores (Tensor[float], [N,]): H[p(y|x_i)] for i in [1, N].
        """
        probs = torch.mean(probs, dim=-1)  # [N, C]
        scores = -torch.sum(torch.xlogy(probs, probs), dim=-1)  # [N,]
        return scores  # [N,]

    @requireprobs
    def compute_score(self, predictions, training_predictions):
        """
        Compute the expected predictive information gain for each candidate input, x_i:
            EPIG(x_i) = E_{p_*(x_*)}[I(y;y_*|x_i,x_*)]
                      = H[p(y|x_i)] + E_{p_*(x_*)}[H[p(y_*|x_*)]] - E_{p_*(x_*)}[H[p(y,y_*|x_i,x_*)]]
        where x_* ~ p_*(x_*) is a target input with unknown label y_*.


        Args:
            predictions (ndarray, [N_p, C, K]): p(y|x_i,θ_j) for i in [1, N_p] and j in [1, K].
            training_predictions (ndarray, [N_t, C, K]): p(y|x_*^i,θ_j) for i in [1, N_t] and j in [1, K].

        Returns:
            scores (ndarray, [N,]): EPIG(x_i) for i in [1, N_p].
        """
        assert predictions.ndim == 3, "EPIG only supports classification for now."
        assert training_predictions.ndim == 3, "EPIG only supports classification for now."

        probs_pool = torch.Tensor(predictions)  # [N_p, C, K]
        probs_targ = torch.Tensor(training_predictions)  # [N_t, C, K]

        N_t, C, K = probs_targ.shape

        entropy_pool = self.marginal_entropy_from_probs(probs_pool)  # [N_p,]
        entropy_targ = self.marginal_entropy_from_probs(probs_targ)  # [N_t,]

        probs_targ = probs_targ.permute(2, 0, 1)  # [K, N_t, C]
        probs_targ = probs_targ.reshape(K, N_t * C)  # [K, N_t * C]
        probs_joint = torch.matmul(probs_pool, probs_targ) / K  # [N_p, C, N_t * C]

        entropy_joint = -torch.sum(torch.xlogy(probs_joint, probs_joint), dim=(-2, -1)) / N_t  # [N_p,]
        entropy_joint = torch.nan_to_num(entropy_joint, nan=0.0)  # [N_p,]

        scores = entropy_pool + torch.mean(entropy_targ) - entropy_joint  # [N_p,]
        return scores.numpy()  # [N_p,]
