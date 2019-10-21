import copy
from contextlib import contextmanager

import torch
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd


class _SeededDropoutNd(_DropoutNd):
    """
    MCDropout layer that can be seeded.

    Notes:
        Using the seeded version is 3x slower than normal Dropout.
    """

    def __init__(self, p=0.5, inplace=False, seed=None):
        super().__init__(p, inplace)
        self.seed = seed
        self.use_cuda = False
        self.gen = None
        self.cpu()

    def _to_right_device(self, is_cuda):
        if is_cuda:
            self.cuda()
        else:
            self.cpu()

    def cuda(self, device=None):
        if not self.use_cuda or self.gen is None:
            self.use_cuda = True
            self.gen = torch.Generator('cuda')
            self.gen.manual_seed(self.seed)
        return super().cuda(device)

    def cpu(self, device=None):
        if self.use_cuda or self.gen is None:
            self.use_cuda = False
            self.gen = torch.Generator()
            self.gen.manual_seed(self.seed)
        return super().cpu()


class SeededDropout(_SeededDropoutNd):
    def forward(self, input):
        self._to_right_device(input.is_cuda)
        with temp_torch_seed(self.gen, self.use_cuda):
            return F.dropout(input, self.p, True, self.inplace)


class SeededDropout2d(_SeededDropoutNd):
    def forward(self, input):
        self._to_right_device(input.is_cuda)
        with temp_torch_seed(self.gen, self.use_cuda):
            return F.dropout2d(input, self.p, True, self.inplace)


def patch_module(module: torch.nn.Module, inplace: bool = True, seed=None) -> torch.nn.Module:
    """Replace dropout layers in a model with MC Dropout layers.

    Parameters
    ----------
    module : torch.nn.Module
        The module in which you would like to replace dropout layers.
    inplace : bool, optional
        Whether to modify the module in place or return a copy of the module.

    Returns
    -------
    torch.nn.Module
        The modified module, which is either the same object as you passed in
        (if inplace = True) or a copy of that object.
    """
    if not inplace:
        module = copy.deepcopy(module)
    _patch_dropout_layers(module, seed=seed)
    return module


def _patch_dropout_layers(module: torch.nn.Module, seed=None) -> None:
    """
    Recursively iterate over the children of a module and replace them if
    they are a dropout layer. This function operates in-place.
    """

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Dropout):
            new_module = SeededDropout(p=child.p, inplace=child.inplace, seed=seed)
        elif isinstance(child, torch.nn.Dropout2d):
            new_module = SeededDropout2d(p=child.p, inplace=child.inplace, seed=seed)
        else:
            new_module = None

        if new_module is not None:
            module.add_module(name, new_module)

        # recursively apply to child
        _patch_dropout_layers(child, seed=seed)


class SeededMCDropoutModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, seed=None):
        """Create a module that with all dropout layers patched.

        Parameters
        ----------
        module : torch.nn.Module
            A fully specified neural network.
        """
        super().__init__()
        self.parent_module = module
        _patch_dropout_layers(self.parent_module, seed=seed)
        self.forward = self.parent_module.forward


@contextmanager
def temp_torch_seed(generator, use_cuda):
    """
    Temporally set the global Torch seed and reset after.
    Args:
        generator (torch.Generator): RNG on the right device.
        use_cuda (bool): Whether to change the CUDA RNG or CPU RNG.
    """
    if use_cuda:
        cuda_state = torch.cuda.get_rng_state()
        gen_state = generator.get_state()
        try:
            yield torch.cuda.set_rng_state(gen_state)
            generator.set_state(torch.cuda.get_rng_state())
        finally:
            torch.cuda.set_rng_state(cuda_state)
    else:

        state = torch.random.get_rng_state()

        gen_state = generator.get_state()
        try:
            yield torch.random.set_rng_state(gen_state)
            generator.set_state(torch.random.get_rng_state())
        finally:
            torch.random.set_rng_state(state)
