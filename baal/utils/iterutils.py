from collections.abc import Sequence, MutableMapping


def map_on_tensor(fn, val):
    """Map a function on a Tensor or a list of Tensors"""
    if isinstance(val, Sequence):
        return [map_on_tensor(fn, v) for v in val]
    elif isinstance(val, (dict, MutableMapping)):
        return {k: map_on_tensor(fn, v) for k, v in val.items()}
    return fn(val)
