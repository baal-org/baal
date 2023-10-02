"""Some heuristics are simply combinations of others."""
from typing import Union, Callable

from baal.active.heuristics import BALD, Entropy
from baal.active.heuristics.stochastics import PowerSampling


def PowerBALD(query_size: int, temperature: float = 1.0, reduction: Union[str, Callable] = "none"):
    return PowerSampling(BALD(reduction=reduction), query_size=query_size, temperature=temperature)


def PowerEntropy(
    query_size: int, temperature: float = 1.0, reduction: Union[str, Callable] = "none"
):
    return PowerSampling(
        Entropy(reduction=reduction), query_size=query_size, temperature=temperature
    )
