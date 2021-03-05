import copy
import warnings
from typing import List

import torch
from packaging.version import parse as parse_version

torch_version = parse_version(torch.__version__)

Sequence = List[str]


def get_weight_drop_module(name: str, weight_dropout, **kwargs):
    return {
        'Conv2d': WeightDropConv2d,
        'Linear': WeightDropLinear
    }[name](weight_dropout, **kwargs)


class WeightDropLinear(torch.nn.Linear):
    """
    Thanks to PytorchNLP for the initial implementation
    # code from https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
    of class WeightDropLinear. Their `License
    <https://github.com/PetrochukM/PyTorch-NLP/blob/master/LICENSE>`__.
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, **kwargs):
        wanted = ['in_features', 'out_features']
        kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        super().__init__(**kwargs)
        self._weight_dropout = weight_dropout

    def forward(self, input):
        w = torch.nn.functional.dropout(self.weight, p=self._weight_dropout, training=True)
        return torch.nn.functional.linear(input, w, self.bias)


class WeightDropConv2d(torch.nn.Conv2d):
    """
    Reimplemmentation of WeightDrop for Conv2D. Thanks to PytorchNLP for the initial implementation
    of class WeightDropLinear. Their `License
    <https://github.com/PetrochukM/PyTorch-NLP/blob/master/LICENSE>`__.
    Wrapper around :class: 'torch.nn.Conv' that adds '' weight_dropout '' named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, **kwargs):
        wanted = ['in_channels', 'out_channels', 'kernel_size', 'dilation', 'padding']
        kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        super().__init__(**kwargs)
        self._weight_dropout = weight_dropout

    def forward(self, input):
        kwargs = {'input': input,
                  'weight': torch.nn.functional.dropout(self.weight, p=self._weight_dropout,
                                                        training=True)}
        if torch_version >= parse_version('1.8.0'):
            # Bias was added as a required argument in this version.
            kwargs['bias'] = self.bias
        return self._conv_forward(**kwargs)


def patch_module(module: torch.nn.Module,
                 layers: Sequence,
                 weight_dropout: float = 0.0,
                 inplace: bool = True) -> torch.nn.Module:
    """Replace given layers with weight_drop module of that layer.

    Args:
        module : torch.nn.Module
            The module in which you would like to replace dropout layers.
        layers : list[str]
            Name of layers to be replaced from ['Conv', 'Linear', 'LSTM', 'GRU'].
        weight_dropout (float): The probability a weight will be dropped.
        inplace : bool, optional
            Whether to modify the module in place or return a copy of the module.

    Returns:
        torch.nn.Module:
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    if not inplace:
        module = copy.deepcopy(module)
    changed = _patch_layers(module, layers, weight_dropout)
    if not changed:
        warnings.warn('No layer was modified by patch_module!', UserWarning)
    return module


def _patch_layers(module: torch.nn.Module, layers: Sequence, weight_dropout: float) -> bool:
    """
    Recursively iterate over the children of a module and replace them if
    they are in the layers list. This function operates in-place.
    """
    changed = False
    for name, child in module.named_children():
        new_module = None
        for layer in layers:
            if isinstance(child, getattr(torch.nn, layer)):
                new_module = get_weight_drop_module(layer, weight_dropout, **child.__dict__)
                break

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # The dropout layer should be deactivated to use DropConnect.
        if isinstance(child, torch.nn.Dropout):
            child.p = 0

        # Recursively apply to child.
        changed = changed or _patch_layers(child, layers, weight_dropout)
    return changed


class MCDropoutConnectModule(torch.nn.Module):
    """ Create a module that with all dropout layers patched.
    With MCDropoutConnectModule, it could be decided which type of modules to be
    replaced.

    Args:
        module (torch.nn.Module):
            A fully specified neural network.
        layers (list[str]):
            Name of layers to be replaced from ['Conv', 'Linear', 'LSTM', 'GRU'].
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, module: torch.nn.Module, layers: Sequence, weight_dropout=0.0):
        super().__init__()
        self.parent_module = module
        _patch_layers(self.parent_module, layers, weight_dropout)

    def forward(self, x):
        return self.parent_module(x)
