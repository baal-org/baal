from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from baal.bayesian.common import BayesianModule, _patching_wrapper


class Dropout(_DropoutNd):
    r"""Randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.
    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .
    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training.
    Shape:
        - Input: :math:`(*)`. Input can be of any shape.
        - Output: :math:`(*)`. Output is of the same shape as input.

    Args:
        p (float, optional):
            Probability of an element to be zeroed. Default: 0.5
        inplace (bool, optional):
            If set to ``True``, will do this operation in-place. Default: ``False``

    Examples::
        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


class Dropout2d(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call.
    with probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Args:
        p (float, optional):
            Probability of an element to be zero-ed.
        inplace (bool, optional):
            If set to ``True``, will do this operation in-place.

    Examples::

        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280

    """

    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)


def patch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    """Replace dropout layers in a model with MCDropout layers.

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
    return _patching_wrapper(module, inplace=inplace, patching_fn=_dropout_mapping_fn)


def unpatch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    """Replace MCDropout layers in a model with Dropout layers.

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
    return _patching_wrapper(module, inplace=inplace, patching_fn=_dropout_unmapping_fn)


def _dropout_mapping_fn(module: torch.nn.Module) -> Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    if isinstance(module, torch.nn.Dropout):
        new_module = Dropout(p=module.p, inplace=module.inplace)
    elif isinstance(module, torch.nn.Dropout2d):
        new_module = Dropout2d(p=module.p, inplace=module.inplace)
    return new_module


def _dropout_unmapping_fn(module: torch.nn.Module) -> Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    if isinstance(module, Dropout):
        new_module = torch.nn.Dropout(p=module.p, inplace=module.inplace)
    elif isinstance(module, Dropout2d):
        new_module = torch.nn.Dropout2d(p=module.p, inplace=module.inplace)
    return new_module


class MCDropoutModule(BayesianModule):
    """Create a module that with all dropout layers patched.

    Args:
        module (torch.nn.Module):
            A fully specified neural network.
    """

    patching_function = patch_module
    unpatch_function = unpatch_module
