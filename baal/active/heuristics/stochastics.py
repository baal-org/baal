import types

import numpy as np
import structlog
from scipy.special import softmax
from scipy.stats import rankdata

from baal.active.heuristics import AbstractHeuristic, Sequence

log = structlog.get_logger(__name__)
EPSILON = 1e-8


class StochasticHeuristic(AbstractHeuristic):
    def __init__(self, base_heuristic: AbstractHeuristic, query_size):
        """Heuristic that is stochastic to improve diversity.

        Common acquisition functions are heavily impacted by duplicates.
        When using a `top-k` approache where the most
        uncertain examples are selected, the acquisition function can select many duplicates.
        Techniques such as BADGE (Ash et al, 2019) or BatchBALD (Kirsh et al. 2019)
        are common solutions to this problem, but they are quite expensive.

        Stochastic acquisitions are cheap to compute and get similar performances.

        References:
            Stochastic Batch Acquisition for Deep Active Learning, Kirsch et al. (2022)
            https://arxiv.org/abs/2106.12059

        Args:
            base_heuristic: Heuristic to get uncertainty from before sampling.
            query_size: These heuristics will return `query_size` items.
        """
        # TODO handle reverse
        super().__init__(reverse=False)
        self._bh = base_heuristic
        self.query_size = query_size

    def get_ranks(self, predictions):
        # Get the raw uncertainty from the base heuristic.
        scores = self.get_scores(predictions)
        # Create the distribution to sample from.
        distributions = self._make_distribution(scores)
        # Force normalization for np.random.choice
        distributions = np.clip(distributions, 0)
        distributions /= distributions.sum()

        # TODO Seed?
        if (distributions > 0).sum() < self.query_size:
            log.warnings("Not enough values, return random")
            distributions = np.ones_like(distributions) / len(distributions)
        return (
            np.random.choice(len(distributions), self.query_size, replace=False, p=distributions),
            distributions,
        )

    def get_scores(self, predictions):
        if isinstance(predictions, types.GeneratorType):
            scores = self._bh.get_uncertainties_generator(predictions)
        else:
            scores = self._bh.get_uncertainties(predictions)
        if isinstance(scores, Sequence):
            scores = np.concatenate(scores)
        return scores

    def _make_distribution(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PowerSampling(StochasticHeuristic):
    def __init__(self, base_heuristic: AbstractHeuristic, query_size, temperature=1.0):
        """Samples from the uncertainty distribution without modification beside
        temperature scaling and normalization.

        Stochastic heuristic that assumes that the uncertainty distribution
        is positive and that items with near-zero uncertainty are uninformative.
        Empirically worked the best in the paper.

        References:
            Stochastic Batch Acquisition for Deep Active Learning, Kirsch et al. (2022)
            https://arxiv.org/abs/2106.12059

        Args:
            base_heuristic: Heuristic to get uncertainty from before sampling.
            query_size: These heuristics will return `query_size` items.
            temperature: Value to temper the uncertainty distribution before sampling.
        """
        super().__init__(base_heuristic=base_heuristic, query_size=query_size)
        self.temperature = temperature

    def _make_distribution(self, scores: np.ndarray) -> np.ndarray:
        scores = scores ** (1 / self.temperature)
        scores = scores / scores.sum()
        return scores


class GibbsSampling(StochasticHeuristic):
    def __init__(self, base_heuristic: AbstractHeuristic, query_size, temperature=1.0):
        """Samples from the uncertainty distribution after applying softmax.

        References:
            Stochastic Batch Acquisition for Deep Active Learning, Kirsch et al. (2022)
            https://arxiv.org/abs/2106.12059

        Args:
            base_heuristic: Heuristic to get uncertainty from before sampling.
            query_size: These heuristics will return `query_size` items.
            temperature: Value to temper the uncertainty distribution before sampling.
        """
        super().__init__(base_heuristic=base_heuristic, query_size=query_size)
        self.temperature = temperature

    def _make_distribution(self, scores: np.ndarray) -> np.ndarray:
        scores /= self.temperature
        # scores dimensions is [N]
        scores = softmax(scores)
        return scores


class RankBasedSampling(StochasticHeuristic):
    def __init__(self, base_heuristic: AbstractHeuristic, query_size, temperature=1.0):
        """Samples from the ranks of the uncertainty distribution.

        References:
            Stochastic Batch Acquisition for Deep Active Learning, Kirsch et al. (2022)
            https://arxiv.org/abs/2106.12059

        Args:
            base_heuristic: Heuristic to get uncertainty from before sampling.
            query_size: These heuristics will return `query_size` items.
            temperature: Value to temper the uncertainty distribution before sampling.
        """
        super().__init__(base_heuristic=base_heuristic, query_size=query_size)
        self.temperature = temperature

    def _make_distribution(self, scores: np.ndarray) -> np.ndarray:
        rank = rankdata(-scores)
        weights = rank ** (-1 / self.temperature)
        normalized_weights: np.ndarray = weights / weights.sum()
        return normalized_weights
