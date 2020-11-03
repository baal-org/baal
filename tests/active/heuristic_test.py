import numpy as np
import pytest
from hypothesis import given
from torch_hypothesis import class_logits

from baal.active import get_heuristic
from baal.active.heuristics import (
    Random,
    BALD,
    Margin,
    Entropy,
    Certainty,
    Variance,
    BatchBALD,
    AbstractHeuristic,
    requireprobs,
    Precomputed,
    CombineHeuristics,
)

N_ITERATIONS = 50
IMG_SIZE = 3
N_CLASS = 10


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i: i + n]


def _make_3d_fake_dist(means, stds, dims=10):
    d = np.stack(
        [_make_fake_dist(means, stds, dims=dims) for _ in range(N_ITERATIONS)]
    )  # 50 iterations
    d = np.rollaxis(d, 0, 3)
    # [n_sample, n_class, n_iter]
    return d


def _make_5d_fake_dist(means, stds, dims=10):
    d = np.stack(
        [_make_3d_fake_dist(means, stds, dims=dims) for _ in range(IMG_SIZE ** 2)], -1
    )  # 3x3 image
    b, c, i, hw = d.shape
    d = np.reshape(d, [b, c, i, IMG_SIZE, IMG_SIZE])
    d = np.rollaxis(d, 2, 5)
    # [n_sample, n_class, H, W, iter]
    return d


def _make_fake_dist(means, stds, dims=10):
    """
    Create some fake discrete distributions
    Args:
        means: List of means
        stds: List of standard deviations
        dims: Dimensions of the distributions

    Returns:
        List of distributions
    """
    n_trials = 100
    distributions = []
    for m, std in zip(means, stds):
        dist = np.zeros([dims])
        for i in range(n_trials):
            dist[
                np.round(np.clip(np.random.normal(m, std, 1), 0, dims - 1)).astype(np.int).item()
            ] += 1
        distributions.append(dist / n_trials)
    return np.array(distributions)


distribution_2d = _make_fake_dist([5, 6, 9], [0.1, 4, 2], dims=N_CLASS)
distributions_3d = _make_3d_fake_dist([5, 6, 9], [0.1, 4, 2], dims=N_CLASS)
distributions_5d = _make_5d_fake_dist([5, 6, 9], [0.1, 4, 2], dims=N_CLASS)


@pytest.mark.parametrize(
    'distributions, reduction',
    [
        (distributions_5d, 'mean'),
        (distributions_5d, lambda x: np.mean(x, axis=(1, 2))),
        (distributions_3d, 'none'),
    ],
)
def test_bald(distributions, reduction):
    np.random.seed(1338)

    bald = BALD(reduction=reduction)
    marg = bald(distributions)
    str_marg = bald(chunks(distributions, 2))

    assert np.allclose(
        bald.get_uncertainties(distributions),
        bald.get_uncertainties_generator(chunks(distributions, 2)),
    )

    assert np.all(marg == [1, 2, 0]), "BALD is not right {}".format(marg)
    assert np.all(str_marg == [1, 2, 0]), "StreamingBALD is not right {}".format(marg)

    bald = BALD(0.99, reduction=reduction)
    marg = bald(distributions)

    # Unlikely, but not 100% sure
    assert np.any(marg != [1, 2, 0])


@pytest.mark.parametrize('distributions, reduction',
                         [(distributions_3d, 'none')])
def test_batch_bald(distributions, reduction):
    np.random.seed(1338)

    bald = BatchBALD(100, reduction=reduction)
    marg = bald(distributions)

    assert np.all(marg[:3] == [1, 2, 0]), "BatchBALD is not right {}".format(marg)

    bald = BatchBALD(100, shuffle_prop=0.99, reduction=reduction)
    marg = bald(distributions)

    # Unlikely, but not 100% sure
    assert np.any(marg != [1, 2, 0])


@pytest.mark.parametrize('distributions, reduction',
                         [(distributions_5d, 'none')])
def test_batch_bald_fails_on_5d(distributions, reduction):
    np.random.seed(1338)

    bald = BatchBALD(100, reduction=reduction)
    with pytest.raises(ValueError) as e:
        marg = bald(distributions)
    assert 'BatchBALD only works on classification' in str(e)


@pytest.mark.parametrize('distributions, reduction',
                         [(distributions_5d, 'mean'),
                          (distributions_5d, lambda x: np.mean(x, axis=(1, 2, 3))),
                          (distributions_3d, 'mean')])
def test_variance(distributions, reduction):
    # WARNING: Highly variable test. (Need more iteration for a better estimation.)
    np.random.seed(1337)

    var = Variance(reduction=reduction)
    marg = var(distributions)
    assert np.all(marg == [1, 2, 0]), "Variance is not right {}".format(marg)

    var = Variance(0.99, reduction=reduction)
    marg = var(distributions)
    assert np.any(marg != [1, 2, 0])


@pytest.mark.parametrize(
    'distributions, reduction',
    [
        (distributions_5d, 'mean'),
        (distributions_5d, lambda x: np.mean(x, axis=(1, 2))),
        (distributions_3d, 'none'),
        (distribution_2d, 'none'),
    ],
)
def test_margin(distributions, reduction):
    np.random.seed(1337)
    margin = Margin(reduction=reduction)
    marg = margin(distributions)
    assert np.all(marg == [1, 2, 0]), "Margin is not right {}".format(marg)

    margin = Margin(shuffle_prop=0.9, reduction=reduction)
    marg = margin(distributions)
    assert np.any(marg != [1, 2, 0])


@pytest.mark.parametrize(
    'distributions, reduction',
    [
        (distributions_5d, 'mean'),
        (distributions_5d, lambda x: np.mean(x, axis=(1, 2))),
        (distributions_3d, 'none'),
        (distribution_2d, 'none'),
    ],
)
def test_entropy(distributions, reduction):
    np.random.seed(1337)
    entropy = Entropy(reduction=reduction)
    marg = entropy(distributions)
    assert np.all(marg == [1, 2, 0]), "Entropy is not right {}".format(marg)

    entropy = Entropy(0.9, reduction=reduction)
    marg = entropy(distributions)
    assert np.any(marg != [1, 2, 0])


@pytest.mark.parametrize(
    'distributions, reduction',
    [
        (distributions_5d, 'mean'),
        (distributions_5d, lambda x: np.mean(x, axis=(1, 2))),
        (distributions_3d, 'none'),
        (distribution_2d, 'none'),
    ],
)
def test_certainty(distributions, reduction):
    np.random.seed(1337)
    certainty = Certainty(reduction=reduction)
    marg = certainty(distributions)
    assert np.all(marg == [1, 2, 0]), "Certainty is not right {}".format(marg)

    certainty = Certainty(0.9, reduction=reduction)
    marg = certainty(distributions)
    assert np.any(marg != [1, 2, 0])


@pytest.mark.parametrize('distributions', [distributions_5d, distributions_3d, distribution_2d])
def test_random(distributions):
    np.random.seed(1337)
    random = Random()
    all_equals = np.all(
        [np.allclose(random(distributions), random(distributions)) for _ in range(10)]
    )
    assert not all_equals


@pytest.mark.parametrize('name', ('random', 'bald', 'variance'))
def test_getting_heuristics(name):
    assert isinstance(get_heuristic(name, reduction='mean'), AbstractHeuristic)


@given(logits=class_logits(batch_size=(1, 64), n_classes=(2, 50)))
def test_that_logits_get_converted_to_probabilities(logits):
    logits = logits.numpy()

    # define a random func:
    @requireprobs
    def wrapped(_, logits):
        return logits

    probability_distribution = wrapped(None, logits)
    assert np.alltrue((probability_distribution >= 0) & (probability_distribution <= 1)).all()


def test_that_precomputed_passes_back_predictions():
    precomputed = Precomputed()
    ranks = np.arange(10)
    assert (precomputed(ranks) == ranks).all()


@pytest.mark.parametrize(
    'heuristic1, heuristic2, weights',
    [(BALD(), Variance(), [0.7, 0.3]),
     (BALD(), Entropy(reduction='mean'), [0.9, 0.8]),
     (Entropy(), Variance(), [4, 8]),
     (Certainty(), Variance(), [9, 2]),
     (Certainty(), Certainty(reduction='mean'), [1, 3])]
)
def test_combine_heuristics(heuristic1, heuristic2, weights):
    np.random.seed(1337)
    predictions = [distributions_3d, distributions_5d]

    if isinstance(heuristic1, Certainty) and not isinstance(heuristic2, Certainty):
        with pytest.raises(Exception) as e_info:
            heuristics = CombineHeuristics([heuristic1, heuristic2], weights=weights,
                                           reduction='mean')
            assert 'heuristics should have the same value for `revesed` parameter' in str(
                e_info.value)
    else:
        heuristics = CombineHeuristics([heuristic1, heuristic2], weights=weights,
                                       reduction='mean')
        if isinstance(heuristic1, Certainty) and isinstance(heuristic2, Certainty):
            assert not heuristics.reversed
        else:
            assert heuristics.reversed
        ranks = heuristics(predictions)
        assert np.all(ranks == [1, 2, 0]), "Combine Heuristics is not right {}".format(ranks)


def test_combine_heuristics_uncertainty_generator():
    np.random.seed(1337)
    prediction_chunks = [chunks(distributions_3d, 2), chunks(distributions_5d, 2)]
    predictions = [distributions_3d, distributions_5d]

    heuristics = CombineHeuristics([BALD(), Variance()], weights=[0.5, 0.5],
                                   reduction='mean')

    assert np.allclose(
        heuristics.get_uncertainties(predictions),
        heuristics.get_uncertainties(prediction_chunks),
    )

    prediction_chunks = [chunks(distributions_3d, 2), chunks(distributions_5d, 2)]
    ranks = heuristics(prediction_chunks)
    assert np.all(ranks == [1, 2, 0]), "Combine Heuristics is not right {}".format(ranks)


def test_heuristics_reorder_list():
    # we are just testing if given calculated uncertainty measures for chunks of data
    # the `reorder_indices` would make correct decision. Here index 0 has the
    # highest uncertainty chosen but both methods (uncertainties1 and uncertainties2)
    streaming_prediction = [np.array([0.98]), np.array([0.87, 0.68]),
                            np.array([0.96, 0.54])]
    heuristic = BALD()
    ranks = heuristic.reorder_indices(streaming_prediction)
    assert np.all(ranks == [0, 3, 1, 2, 4]), "reorder list for BALD is not right {}".format(ranks)

    heuristic = Variance()
    ranks = heuristic.reorder_indices(streaming_prediction)
    assert np.all(ranks == [0, 3, 1, 2, 4]), "reorder list for Variance is not right {}".format(
        ranks)

    heuristic = Entropy()
    ranks = heuristic.reorder_indices(streaming_prediction)
    assert np.all(ranks == [0, 3, 1, 2, 4]), "reorder list for Entropy is not right {}".format(
        ranks)

    heuristic = Margin()
    ranks = heuristic.reorder_indices(streaming_prediction)
    assert np.all(ranks == [4, 2, 1, 3, 0]), "reorder list for Margin is not right {}".format(ranks)

    heuristic = Certainty()
    ranks = heuristic.reorder_indices(streaming_prediction)
    assert np.all(ranks == [4, 2, 1, 3, 0]), "reorder list for Certainty is not right {}".format(
        ranks)

    heuristic = Random()
    ranks = heuristic.reorder_indices(streaming_prediction)
    assert ranks.size == 5, "reorder list for Random is not right {}".format(
        ranks)


def test_combine_heuristics_reorder_list():
    # we are just testing if given calculated uncertainty measures for chunks of data
    # the `reorder_indices` would make correct decision. Here index 0 has the
    # highest uncertainty chosen but both methods (uncertainties1 and uncertainties2)
    bald_firstchunk = np.array([0.98])
    bald_secondchunk = np.array([0.87, 0.68])

    variance_firstchunk = np.array([0.76])
    variance_secondchunk = np.array([0.63, 0.48])
    streaming_prediction = [[bald_firstchunk, variance_firstchunk],
                            [bald_secondchunk, variance_secondchunk]]

    heuristics = CombineHeuristics([BALD(), Variance()], weights=[0.5, 0.5],
                                   reduction='mean')
    ranks = heuristics.reorder_indices(streaming_prediction)
    assert np.all(ranks == [0, 1, 2]), "Combine Heuristics is not right {}".format(ranks)


@pytest.mark.parametrize("heur", [Random(), BALD(reduction='sum'),
                                  Entropy(reduction='sum'),
                                  Variance(reduction='sum')])
@pytest.mark.parametrize("n_batch", [1, 10, 20])
def test_heuristics_works_with_generator(heur, n_batch):
    BATCH_SIZE = 32

    def predictions(n_batch):
        for _ in range(n_batch):
            yield np.random.randn(BATCH_SIZE, 3, 32, 32, 10)

    preds = predictions(n_batch)
    out = heur(preds)
    assert out.shape[0] == n_batch * BATCH_SIZE


@pytest.mark.parametrize('distributions', [distributions_5d])
def test_heuristic_reductio_check(distributions):
    np.random.seed(1337)
    heuristic = BALD(reduction='none')
    with pytest.raises(ValueError) as e_info:
        heuristic(distributions)
        assert "Can't order sequence with more than 1 dimension." in str(e_info.value)


if __name__ == '__main__':
    pytest.main()
