import copy
import warnings
from typing import List, Optional, cast, Dict
from baal.bayesian.common import replace_layers_in_module, _patching_wrapper, BayesianModule
import torch
from torch import nn
from packaging.version import parse as parse_version


Sequence = List[str]


def get_weight_drop_module(name: str, weight_dropout, **kwargs):
    return {"Conv2d": WeightDropConv2d, "Linear": WeightDropLinear}[name](weight_dropout, **kwargs)


class WeightDropMixin:
    _kwargs: Dict

    def unpatch(self):
        new_module = self.__class__.__bases__[0](**self._kwargs)
        new_module.load_state_dict(self.state_dict())
        return new_module


class WeightDropLinear(torch.nn.Linear, WeightDropMixin):
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
        wanted = ["in_features", "out_features"]
        self._kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        super().__init__(**self._kwargs)
        self._weight_dropout = weight_dropout

    def forward(self, input):
        w = torch.nn.functional.dropout(self.weight, p=self._weight_dropout, training=True)
        return torch.nn.functional.linear(input, w, self.bias)


class WeightDropConv2d(torch.nn.Conv2d, WeightDropMixin):
    """
    Reimplemmentation of WeightDrop for Conv2D. Thanks to PytorchNLP for the initial implementation
    of class WeightDropLinear. Their `License
    <https://github.com/PetrochukM/PyTorch-NLP/blob/master/LICENSE>`__.
    Wrapper around :class: 'torch.nn.Conv' that adds '' weight_dropout '' named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, **kwargs):
        wanted = ["in_channels", "out_channels", "kernel_size", "dilation", "padding"]
        self._kwargs = {k: v for k, v in kwargs.items() if k in wanted}
        super().__init__(**self._kwargs)
        self._weight_dropout = weight_dropout
        self._torch_version = parse_version(torch.__version__)

    def forward(self, input):
        kwargs = {
            "input": input,
            "weight": torch.nn.functional.dropout(
                self.weight, p=self._weight_dropout, training=True
            ),
        }
        if self._torch_version >= parse_version("1.8.0"):
            # Bias was added as a required argument in this version.
            kwargs["bias"] = self.bias
        return self._conv_forward(**kwargs)


def patch_module(
    module: torch.nn.Module, layers: Sequence, weight_dropout: float = 0.0, inplace: bool = True
) -> torch.nn.Module:
    """
    Replace given layers with weight_drop module of that layer.

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
    return _patching_wrapper(
        module,
        inplace=inplace,
        patching_fn=_dropconnect_mapping_fn,
        layers=layers,
        weight_dropout=weight_dropout,
    )


def unpatch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    """
    Unpatch Dropconnect module to recover initial module.

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
    return _patching_wrapper(module, inplace=inplace, patching_fn=_droconnect_unmapping_fn)


def _dropconnect_mapping_fn(module: torch.nn.Module, layers, weight_dropout) -> Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    for layer in layers:
        if isinstance(module, getattr(torch.nn, layer)):
            new_module = get_weight_drop_module(layer, weight_dropout, **module.__dict__)
            break
    if isinstance(module, nn.Dropout):
        module._baal_p: float = module.p  # type: ignore
        module.p = 0.0
    return new_module


def _droconnect_unmapping_fn(module: torch.nn.Module) -> Optional[nn.Module]:
    new_module: Optional[nn.Module] = None
    if isinstance(module, WeightDropMixin):
        new_module = module.unpatch()

    if isinstance(module, nn.Dropout):
        module.p = module._baal_p  # type: ignore

    return new_module


class MCDropoutConnectModule(BayesianModule):
    """Create a module that with all dropout layers patched.
    With MCDropoutConnectModule, it could be decided which type of modules to be
    replaced.

    Args:
        module (torch.nn.Module):
            A fully specified neural network.
        layers (list[str]):
            Name of layers to be replaced from ['Conv', 'Linear', 'LSTM', 'GRU'].
        weight_dropout (float): The probability a weight will be dropped.
    """

    patching_function = patch_module
    unpatch_function = unpatch_module
