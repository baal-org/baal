from typing import Optional

import torch
from torch import nn, Tensor

from baal.bayesian.common import BayesianModule, _patching_wrapper


class LRUCacheModule(nn.Module):
    def __init__(self, module, size=1):
        super().__init__()
        if size != 1:
            raise ValueError("We do not support LRUCache bigger than 1.")
        self.module = module
        self._memory_input = None
        self._memory_output = None

    def _is_cache_void(self, x):
        return self._memory_input is None or not torch.equal(self._memory_input, x)

    def __call__(self, x: Tensor):
        if self.training:
            return self.module(x)
        if self._is_cache_void(x):
            self._memory_input = x
            self._memory_output = self.module(x)
        return self._memory_output


def _caching_mapping_fn(module: torch.nn.Module) -> Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    # Could add more
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        new_module = LRUCacheModule(module=module)
    return new_module


def _caching_unmapping_fn(module: torch.nn.Module) -> Optional[nn.Module]:
    new_module: Optional[nn.Module] = None

    if isinstance(module, LRUCacheModule):
        new_module = module.module
    return new_module


def patch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    return _patching_wrapper(module, inplace=inplace, patching_fn=_caching_mapping_fn)


def unpatch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    return _patching_wrapper(module, inplace=inplace, patching_fn=_caching_unmapping_fn)


class MCCachingModule(BayesianModule):
    patching_function = patch_module
    unpatch_function = unpatch_module
