import pytest
import torch
from torch import nn
from torch.nn.modules import Flatten

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


@pytest.fixture()
def dummy_cnn_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.Dropout2d(),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(32, 16),
        nn.Linear(16, 10)
    )


@pytest.mark.parametrize('model, input_shape',
                         [['mlp', [10, 32]], ['cnn', [10, 3, 32, 32]]])
def test_dropout_with_seed(model, input_shape, dummy_model, dummy_cnn_model):
    model = {'mlp': dummy_model, 'cnn': dummy_cnn_model}[model]
    model1 = seeded_dropout.patch_module(model, inplace=False, seed=1337)
    model2 = seeded_dropout.patch_module(model, inplace=False, seed=1337)
    inp = torch.randn(*input_shape)

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
@pytest.mark.parametrize('model, input_shape',
                         [['mlp', [10, 32]], ['cnn', [10, 3, 32, 32]]])
def test_gpu_dropout_with_seed(model, input_shape, dummy_model, dummy_cnn_model):
    model = {'mlp': dummy_model, 'cnn': dummy_cnn_model}[model]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model1 = seeded_dropout.patch_module(model, inplace=False, seed=1337)
    model1 = model1.cuda()
    model2 = seeded_dropout.patch_module(model, inplace=False, seed=1337)
    model2 = model2.cuda()
    inp = torch.randn(*input_shape).cuda()

    model1.train()
    model2.train()
    assert torch.equal(model1(inp), model2(inp))
    assert not torch.equal(model1(inp), model1(inp))
    assert not torch.equal(model2(inp), model2(inp))  # Sync them up again.

    model1.eval()
    model2.eval()
    assert torch.equal(model1(inp), model2(inp))
    assert not torch.equal(model1(inp), model1(inp))


@pytest.mark.parametrize('model, input_shape',
                         [['mlp', [10, 32]], ['cnn', [10, 3, 32, 32]]])
def test_dropout_module(model, input_shape, dummy_model, dummy_cnn_model):
    model = {'mlp': dummy_model, 'cnn': dummy_cnn_model}[model]
    model1 = seeded_dropout.SeededMCDropoutModule(model, 1337)
    inp = torch.randn(*input_shape)
    assert not torch.equal(model1(inp), model1(inp))
