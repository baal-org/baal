import numpy as np

N_ITERATIONS = 50
IMG_SIZE = 3


def make_3d_fake_dist(means, stds, dims=10):
    d = np.stack(
        [make_fake_dist(means, stds, dims=dims) for _ in range(N_ITERATIONS)]
    )  # 50 iterations
    d = np.rollaxis(d, 0, 3)
    # [n_sample, n_class, n_iter]
    return d


def make_5d_fake_dist(means, stds, dims=10):
    d = np.stack(
        [make_3d_fake_dist(means, stds, dims=dims) for _ in range(IMG_SIZE ** 2)], -1
    )  # 3x3 image
    b, c, i, hw = d.shape
    d = np.reshape(d, [b, c, i, IMG_SIZE, IMG_SIZE])
    d = np.rollaxis(d, 2, 5)
    # [n_sample, n_class, H, W, iter]
    return d


def make_fake_dist(means, stds, dims=10):
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
                np.round(np.clip(np.random.normal(m, std, 1), 0, dims - 1)).astype(int).item()
            ] += 1
        distributions.append(dist / n_trials)
    return np.array(distributions)
