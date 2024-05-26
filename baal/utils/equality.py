from typing import Sequence, Mapping, Optional, TypeVar

import numpy as np
import torch
from torch import Tensor

T = TypeVar("T")


def deep_check(obj1, obj2) -> bool:
    if type(obj1) != type(obj2):
        return False
    elif isinstance(obj1, str):
        return bool(obj1 == obj2)
    elif isinstance(obj1, Sequence):
        return all(deep_check(i1, i2) for i1, i2 in zip(obj1, obj2))
    elif isinstance(obj1, Mapping):
        return all(deep_check(val1, obj2[key1]) for key1, val1 in obj1.items())
    elif isinstance(obj1, Tensor):
        return torch.equal(obj1, obj2)
    elif isinstance(obj1, np.ndarray):
        return bool((obj1 == obj2).all())
    else:
        return bool(obj1 == obj2)


def assert_not_none(val: Optional[T]) -> T:
    """
    This function makes sure that the variable is not None and has a fixed type for mypy purposes.
    Args:
        val: any value which is Optional.
    Returns:
        val [T]: The same value with a defined type.
    Raises:
        Assertion error if val is None.
    """
    if val is None:
        raise AssertionError(f"value of {val} is None, expected not None")
    return val
