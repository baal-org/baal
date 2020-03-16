from typing import List
import copy

from torch.nn import Parameter
import torch

Sequence = List[str]


def get_weight_drop_module(name: str, weight_dropout, **kwargs):
    return {
        'Conv2d': WeightDropConv2d,
        'Linear': WeightDropLinear
    }[name](weight_dropout, **kwargs)


# Code from https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html
def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')

            # dropout should work in inference time as well
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=True)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


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
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)


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
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)


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
    _patch_layers(module, layers, weight_dropout)
    return module


def _patch_layers(module: torch.nn.Module, layers: Sequence, weight_dropout: float) -> None:
    """
    Recursively iterate over the children of a module and replace them if
    they are in the layers list. This function operates in-place.
    """

    for name, child in module.named_children():
        new_module = None
        for layer in layers:
            if isinstance(child, getattr(torch.nn, layer)):
                new_module = get_weight_drop_module(layer, weight_dropout, **child.__dict__)
                break

        if new_module is not None:
            module.add_module(name, new_module)

        # The dropout layer should be deactivated to use DropConnect.
        if isinstance(child, torch.nn.Dropout):
            child.p = 0

        # Recursively apply to child.
        _patch_layers(child, layers, weight_dropout)


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
        self.forward = self.parent_module.forward
