import numpy as np
import pytest
from scipy.stats import entropy

from baal.active.heuristics import BALD, Entropy
from baal.active.heuristics.stochastics import GibbsSampling, RankBasedSampling, PowerSampling
from tests.test_utils import make_fake_dist


@pytest.fixture
def sampled_predictions():
    return np.stack([make_fake_dist([1, 2, 2], [1, 3, 3], dims=20) for _ in range(10)])


@pytest.mark.parametrize("stochastic_heuristic", [GibbsSampling, RankBasedSampling, PowerSampling])
@pytest.mark.parametrize("base_heuristic", [BALD, Entropy])
def test_stochastic_heuristic(stochastic_heuristic, base_heuristic, sampled_predictions):
    heur_temp_1 = stochastic_heuristic(base_heuristic(), query_size=100, temperature=1.0)
    heur_temp_10 = stochastic_heuristic(base_heuristic(), query_size=100, temperature=10.0)

    scores = heur_temp_1.get_scores(sampled_predictions)

    dist_temp_1, dist_temp_10 = heur_temp_1._make_distribution(scores), heur_temp_10._make_distribution(scores)

    assert entropy(dist_temp_1) > entropy(dist_temp_10)
