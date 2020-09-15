import copy
import warnings

import torch
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd


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
    if not inplace:
        module = copy.deepcopy(module)
    changed = _patch_dropout_layers(module)
    if not changed:
        warnings.warn('No layer was modified by patch_module!', UserWarning)
    return module


def _patch_dropout_layers(module: torch.nn.Module) -> bool:
    """
    Recursively iterate over the children of a module and replace them if
    they are a dropout layer. This function operates in-place.
    """
    changed = False
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Dropout):
            new_module = ConsistentDropout(p=child.p)
        elif isinstance(child, torch.nn.Dropout2d):
            new_module = ConsistentDropout2d(p=child.p)
        else:
            new_module = None

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # recursively apply to child
        changed = changed or _patch_dropout_layers(child)
    return changed


class MCConsistentDropoutModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """Create a module that with all dropout layers patched.

        Args:
            module (torch.nn.Module):
                A fully specified neural network.
        """
        super().__init__()
        self.parent_module = module
        _patch_dropout_layers(self.parent_module)
        self.forward = self.parent_module.forward
