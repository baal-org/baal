from typing import Union, Callable

from . import heuristics
from .active_loop import ActiveLearningLoop
from .dataset import ActiveLearningDataset
from .file_dataset import FileDataset


def get_heuristic(
    name: str, shuffle_prop: float = 0.0, reduction: Union[str, Callable] = "none", **kwargs
) -> heuristics.AbstractHeuristic:
    """
    Create an heuristic object from the name.

    Args:
        name (str): Name of the heuristic.
        shuffle_prop (float): Shuffling proportion when getting ranks.
        reduction (Union[str, Callable]): Reduction used after computing the score.
        kwargs (dict): Complementary arguments.

    Returns:
        AbstractHeuristic object.
    """
    heuristic: heuristics.AbstractHeuristic = {
        "random": heuristics.Random,
        "certainty": heuristics.Certainty,
        "entropy": heuristics.Entropy,
        "margin": heuristics.Margin,
        "bald": heuristics.BALD,
        "variance": heuristics.Variance,
        "precomputed": heuristics.Precomputed,
        "batch_bald": heuristics.BatchBALD,
        "epig": heuristics.EPIG,
    }[name](shuffle_prop=shuffle_prop, reduction=reduction, **kwargs)
    return heuristic
