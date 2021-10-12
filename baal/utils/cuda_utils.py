from collections.abc import Mapping, Sequence
from functools import singledispatch

import torch


@singledispatch
def to_cuda(data):
    """
    Move an object to CUDA.

    This function works recursively on lists and dicts, moving the values
    inside to cuda.

    Args:
        data (list, tuple, dict, torch.Tensor, torch.nn.Module):
            The data you'd like to move to the GPU. If there's a pytorch tensor or
            model in data (e.g. in a list or as values in a dictionary) this
            function will move them all to CUDA and return something that matches
            the input in structure.

    Returns:
        list, tuple, dict, torch.Tensor, torch.nn.Module:
            Data of the same type / structure as the input.
    """
    # the base case: if this is not a type we recognise, return it
    return data


@to_cuda.register(torch.Tensor)
@to_cuda.register(torch.nn.Module)
def _to_cuda_modules(data):
    return data.cuda()


@to_cuda.register
def _to_cuda_mapping(data: Mapping):
    # use the type of the object to create a new one:
    return type(data)([(key, to_cuda(val)) for key, val in data.items()])  # type: ignore


@to_cuda.register
def _to_cuda_sequence(data: Sequence):
    """
    Move an object to CUDA.

    This function works recursively on lists and dicts, moving the values
    inside to cuda.

    Args:
        data (Sequence):
            The data you'd like to move to the GPU. If there's a pytorch tensor or
            model in data (e.g. in a list or as values in a dictionary) this
            function will move them all to CUDA and return something that matches
            the input in structure.

    Returns:
        Sequence, with the elements converted to cuda if possible
    """
    # use the type of this object to instantiate a new one:
    if hasattr(data, "_fields"):  # in case it's a named tuple
        return type(data)(*(to_cuda(item) for item in data))
    elif isinstance(data, str):
        # Special case
        return data
    else:
        return type(data)(to_cuda(item) for item in data)  # type: ignore
