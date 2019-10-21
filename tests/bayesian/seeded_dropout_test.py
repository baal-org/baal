import pytest
import torch
from torch import nn

from baal.bayesian import seeded_dropout

gpu_available = torch.cuda.is_available()


@pytest.fixture()
def dummy_model():
    return nn.Sequential(
        nn.Linear(32, 32),
        nn.Dropout(),
        nn.Linear(32, 16),
        nn.Dropout(),
        nn.Linear(16, 10)
    )


def test_dropout_with_seed(dummy_model):
    model1 = seeded_dropout.patch_module(dummy_model, inplace=False, seed=1337)
    model2 = seeded_dropout.patch_module(dummy_model, inplace=False, seed=1337)
    inp = torch.randn(10, 32)

    model1.train()
    model2.train()
    assert torch.equal(model1(inp), model2(inp))
    assert not torch.equal(model1(inp), model1(inp))
    assert not torch.equal(model2(inp), model2(inp))  # Sync them up again.

    model1.eval()
    model2.eval()
    assert torch.equal(model1(inp), model2(inp))
    assert not torch.equal(model1(inp), model1(inp))


@pytest.mark.skipif(not gpu_available, reason='Need gpu')
def test_gpu_dropout_with_seed(dummy_model):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model1 = seeded_dropout.patch_module(dummy_model, inplace=False, seed=1337)
    model1 = model1.cuda()
    model2 = seeded_dropout.patch_module(dummy_model, inplace=False, seed=1337)
    model2 = model2.cuda()
    inp = torch.randn(10, 32).cuda()

    model1.train()
    model2.train()
    assert torch.equal(model1(inp), model2(inp))
    assert not torch.equal(model1(inp), model1(inp))
    assert not torch.equal(model2(inp), model2(inp))  # Sync them up again.

    model1.eval()
    model2.eval()
    assert torch.equal(model1(inp), model2(inp))
    assert not torch.equal(model1(inp), model1(inp))
