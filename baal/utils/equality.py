from typing import Sequence, Mapping

import numpy as np
import torch
from torch import Tensor


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
