from collections.abc import Sequence


def map_on_tensor(fn, val):
    """Map a function on a Tensor or a list of Tensors"""
    if isinstance(val, Sequence):
        return [fn(v) for v in val]
    return fn(val)


def map_on_dict(fn, val):
    """Map a function on a Tensor or a list of Tensors"""
    if isinstance(val, dict):
        return {k: fn(v) for k, v in val.items()}
    return fn(val)
