from typing import Callable
from torch import nn


def replace_layers_in_module(module: nn.Module, mapping_fn: Callable) -> bool:
    """
    Recursively iterate over the children of a module and replace them according to `mapping_fn`.

    Returns:
        True if a layer has been changed.
    """
    changed = False
    for name, child in module.named_children():
        new_module = mapping_fn(child)

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # recursively apply to child
        changed |= replace_layers_in_module(child, mapping_fn)
    return changed
