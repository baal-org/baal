import copy
import warnings
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from baal.bayesian.common import replace_layers_in_module, _patching_wrapper, BayesianModule


class ConsistentDropout(_DropoutNd):
    """
    ConsistentDropout is useful when doing research.
    It guarantees that while the masks are the same between batches
    during inference. The masks are different inside the batch.

    This is slower than using regular Dropout, but it is useful
    when you want to use the same set of weights for each sample used in inference.

    From BatchBALD (Kirsch et al, 2019), this is necessary to use BatchBALD and remove noise
    from the prediction.

    Args:
        p (float): probability of an element to be zeroed. Default: 0.5

    Notes:
        For optimal results, you should use a batch size of one
        during inference time.
        Furthermore, to guarantee that each sample uses the same
        set of weights,
        you must use `replicate_in_memory=True` in ModelWrapper,
        which is the default.
    """

    def __init__(self, p=0.5):
        super().__init__(p=p, inplace=False)
        self.reset()

    def forward(self, x):
        if self.training:
            return F.dropout(x, self.p, training=True, inplace=False)
        else:
            if self._mask is None or self._mask.shape != x.shape:
                self._mask = self._make_mask(x)
            return torch.mul(x, self._mask)

    def _make_mask(self, x):
        return F.dropout(torch.ones_like(x, device=x.device), self.p, training=True)

    def reset(self):
        self._mask = None

    def eval(self):
        self.reset()
        return super().eval()


class ConsistentDropout2d(_DropoutNd):
    """
    ConsistentDropout is useful when doing research.
    It guarantees that while the mask are the same between batches,
    they are different inside the batch.

    This is slower than using regular Dropout, but it is useful
    when you want to use the same set of weights for each unlabelled sample.

    Args:
        p (float): probability of an element to be zeroed. Default: 0.5

    Notes:
        For optimal results, you should use a batch size of one
        during inference time.
        Furthermore, to guarantee that each sample uses the same
        set of weights,
        you must use `replicate_in_memory=True` in ModelWrapper,
        which is the default.
    """

    def __init__(self, p=0.5):
        super().__init__(p=p, inplace=False)
        self.reset()

    def forward(self, x):
        if self.training:
            return F.dropout2d(x, self.p, training=True, inplace=False)
        else:
            if self._mask is None or self._mask.shape != x.shape:
                self._mask = self._make_mask(x)
            return torch.mul(x, self._mask)

    def _make_mask(self, x):
        return F.dropout2d(torch.ones_like(x, device=x.device), self.p, training=True)

    def reset(self):
        self._mask = None

    def eval(self):
        self.reset()
        return super().eval()


def patch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    """Replace dropout layers in a model with Consistent Dropout layers.

    Args:
        module (torch.nn.Module):
            The module in which you would like to replace dropout layers.
        inplace (bool, optional):
            Whether to modify the module in place or return a copy of the module.

    Returns:
        torch.nn.Module
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    return _patching_wrapper(module, inplace=inplace, patching_fn=_consistent_dropout_mapping_fn)


def unpatch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    """Replace ConsistentDropout layers in a model with Dropout layers.

    Args:
        module (torch.nn.Module):
            The module in which you would like to replace dropout layers.
        inplace (bool, optional):
            Whether to modify the module in place or return a copy of the module.

    Returns:
        torch.nn.Module
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    return _patching_wrapper(module, inplace=inplace, patching_fn=_consistent_dropout_unmapping_fn)


def _consistent_dropout_mapping_fn(module: torch.nn.Module) -> Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    if isinstance(module, torch.nn.Dropout):
        new_module = ConsistentDropout(p=module.p)
    elif isinstance(module, torch.nn.Dropout2d):
        new_module = ConsistentDropout2d(p=module.p)
    return new_module


def _consistent_dropout_unmapping_fn(module: torch.nn.Module) -> Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    if isinstance(module, ConsistentDropout):
        new_module = torch.nn.Dropout(p=module.p)
    elif isinstance(module, ConsistentDropout2d):
        new_module = torch.nn.Dropout2d(p=module.p)
    return new_module


class MCConsistentDropoutModule(BayesianModule):
    patching_function = patch_module
    unpatch_function = unpatch_module
