import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture
def is_deterministic():
    def fn(module: nn.Module, input_shape):
        inp = torch.randn(*input_shape)
        pred1 = module(inp).detach().cpu().numpy()
        return all(np.allclose(pred1, module(inp).detach().cpu().numpy()) for _ in range(5))
    return fn


@pytest.fixture
def sampled_predictions():
    return np.random.randn(100, 10, 20)