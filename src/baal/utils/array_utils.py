import numpy as np
from scipy.special import softmax


def to_prob(probabilities: np.ndarray):
    """
    If the probabilities array is not a distrubution will softmax it.

    Args:
        probabilities (array): [batch_size, num_classes, ...]

    Returns:
        Same as probabilities.
    """
    bounded = np.min(probabilities) < 0 or np.max(probabilities) > 1.0
    if bounded or not np.allclose(probabilities.sum(1), 1):
        probabilities = softmax(probabilities, 1)
    return probabilities
