import numpy as np
import pytest

from hypothesis import given, assume, strategies as st
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
)

N_ITERATIONS = 50
IMG_SIZE = 3
N_CLASS = 10


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


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

    bald = BALD(threshold=0.1, reduction=reduction)
    marg = bald(distributions)
    assert np.any(distributions[marg] <= 0.1)

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

    assert np.all(marg == [1, 2, 0][:len(marg)]), "BatchBALD is not right {}".format(marg)

    bald = BatchBALD(100, threshold=0.1, reduction=reduction)
    marg = bald(distributions)
    assert np.any(distributions[marg] <= 0.1)

    bald = BatchBALD(100, 0.99, reduction=reduction)
    marg = bald(distributions)

    # Unlikely, but not 100% sure
    assert np.any(marg != [1, 2, 0])


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

    var = Variance(threshold=0.1, reduction=reduction)
    marg = var(distributions)
    assert np.any(distributions[marg] <= 0.1)

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

    margin = Margin(threshold=0.1, reduction=reduction)
    marg = margin(distributions)
    assert np.any(distributions[marg] <= 0.1)


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

    entropy = Entropy(threshold=0.1, reduction=reduction)
    marg = entropy(distributions)
    assert np.any(distributions[marg] <= 0.1)


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

    certainty = Certainty(threshold=0.1, reduction=reduction)
    marg = certainty(distributions)
    assert np.any(distributions[marg] <= 0.1)


@pytest.mark.parametrize('distributions', [distributions_5d, distributions_3d, distribution_2d])
def test_random(distributions):
    np.random.seed(1337)
    random = Random()
    all_equals = np.all(
        [np.allclose(random(distributions), random(distributions)) for _ in range(10)]
    )
    assert not all_equals

    random = Random(threshold=0.1)
    marg = random(distributions)
    assert np.any(distributions[marg] <= 0.1)


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


if __name__ == '__main__':
    pytest.main()
