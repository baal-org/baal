import numpy as np
import pytest
from scipy.stats import entropy

from baal.active.heuristics import BALD, Entropy
from baal.active.heuristics.stochastics import GibbsSampling, RankBasedSampling, PowerSampling

NUM_CLASSES = 10
NUM_ITERATIONS = 20
BATCH_SIZE = 32


@pytest.fixture
def sampled_predictions():
    predictions = np.stack(
        [np.histogram(np.random.rand(5), bins=np.linspace(-.5, .5, NUM_CLASSES + 1))[0] for _ in
         range(BATCH_SIZE * NUM_ITERATIONS)]).reshape(
        [BATCH_SIZE, NUM_ITERATIONS, NUM_CLASSES])
    return np.rollaxis(predictions, -1, 1)


@pytest.mark.parametrize("stochastic_heuristic", [GibbsSampling, RankBasedSampling, PowerSampling])
@pytest.mark.parametrize("base_heuristic", [BALD, Entropy])
def test_stochastic_heuristic(stochastic_heuristic, base_heuristic, sampled_predictions):
    heur_temp_1 = stochastic_heuristic(base_heuristic(), query_size=100, temperature=1.0)
    heur_temp_10 = stochastic_heuristic(base_heuristic(), query_size=100, temperature=10.0)
    heur_temp_05 = stochastic_heuristic(base_heuristic(), query_size=100, temperature=0.01)

    scores = heur_temp_1.get_scores(sampled_predictions)

    dist_temp_1, dist_temp_10, dist_temp_05 = (heur_temp_1._make_distribution(scores),
                                               heur_temp_10._make_distribution(scores),
                                               heur_temp_05._make_distribution(scores))

    assert entropy(dist_temp_1) < entropy(dist_temp_10)
    # NOTE: it is possible that this fails, as temp_1 can already have minimal entropy. This is unlikely.
    assert entropy(dist_temp_1) > entropy(dist_temp_05)
