from functools import singledispatch
from collections.abc import Mapping, Sequence
import torch


@singledispatch
def to_cuda(data):
    """Move an object to CUDA.

    This function works recursively on lists and dicts, moving the values
    inside to cuda.

    Parameters
    ----------
    data : list, dict, torch.Tensor, torch.nn.Module, 
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # the base case: if this is not a type we recognise, return it
    return data


@to_cuda.register(torch.Tensor)
@to_cuda.register(torch.nn.Module)
def _(x):
    return x.cuda()


@to_cuda.register
def _(x: Mapping):
    # use the type of the object to create a new one:
    return type(x)([(key, to_cuda(val)) for key, val in x.items()])


@to_cuda.register
def _(x: Sequence):
    # use the type of this object to instantiate a new one:
    if hasattr(x, "_fields"):  # in case it's a named tuple
        return type(x)(*(to_cuda(item) for item in x))
    else:
        return type(x)(to_cuda(item) for item in x)
