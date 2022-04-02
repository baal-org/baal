from typing import Callable

import torch
from torch import nn


def replace_layers_in_module(module: nn.Module, mapping_fn: Callable, *args, **kwargs) -> bool:
    """
    Recursively iterate over the children of a module and replace them according to `mapping_fn`.

    Returns:
        True if a layer has been changed.
    """
    changed = False
    for name, child in module.named_children():
        new_module = mapping_fn(child, *args, **kwargs)

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # recursively apply to child
        changed |= replace_layers_in_module(child, mapping_fn, *args, **kwargs)
    return changed


class BayesianModule(torch.nn.Module):
    patching_function: Callable[..., torch.nn.Module]
    unpatch_function: Callable[..., torch.nn.Module]

    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.parent_module = self.__class__.patching_function(module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.parent_module(*args, **kwargs)

    def unpatch(self) -> torch.nn.Module:
        return self.__class__.unpatch_function(self.parent_module)

    # Context Manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unpatch()
