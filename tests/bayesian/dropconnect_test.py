import warnings

import pytest
import torch

from baal.bayesian.weight_drop import patch_module, WeightDropLinear, MCDropoutConnectModule, WeightDropConv2d


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=10)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(8, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


@pytest.mark.parametrize("inplace", (True, False))
@pytest.mark.parametrize("layers", (['Linear'], ['Linear', 'Conv2d'], ['Conv2d']))
def test_patch_module_changes_weights(inplace, layers, is_deterministic):
    test_module = SimpleModel()
    test_module.eval()
    simple_input = torch.randn(10, 3, 10, 10)
    assert torch.allclose(test_module(simple_input), test_module(simple_input))

    mc_test_module = patch_module(test_module, layers=layers, weight_dropout=0.2, inplace=inplace)

    # objects should be the same if inplace is True and not otherwise:
    assert (mc_test_module is test_module) == inplace
    assert not is_deterministic(mc_test_module, (10, 3, 10, 10))

    assert list(mc_test_module.modules())[3].p == 0


@pytest.mark.parametrize("inplace", (True, False))
@pytest.mark.parametrize("layers", (['Conv2d'],))
def test_patch_module_raise_warnings(inplace, layers):
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    with warnings.catch_warnings(record=True) as w:
        mc_test_module = patch_module(test_module, layers=layers,
                                      weight_dropout=0.2, inplace=inplace)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "No layer was modified by patch_module" in str(w[-1].message)


@pytest.mark.parametrize("inplace", (True, False))
def test_patch_module_replaces_all_dropout_layers(inplace):
    test_module = SimpleModel()

    mc_test_module = patch_module(test_module, inplace=inplace, layers=['Conv2d', 'Linear', 'LSTM', 'GRU'])

    # objects should be the same if inplace is True and not otherwise:
    assert (mc_test_module is test_module) == inplace
    assert not any(
        module.p != 0 for module in mc_test_module.modules() if isinstance(module, torch.nn.Dropout)
    )
    assert any(
        isinstance(module, WeightDropLinear)
        for module in mc_test_module.modules()
    )


def test_mcdropconnect_replaces_all_dropout_layers_module(is_deterministic):
    test_module = SimpleModel()

    mc_test_module = MCDropoutConnectModule(test_module, layers=['Conv2d', 'Linear', 'LSTM', 'GRU'], weight_dropout=0.5)

    assert not any(
        module.p != 0 for module in mc_test_module.modules() if isinstance(module, torch.nn.Dropout)
    )
    assert any(
        isinstance(module, WeightDropLinear)
        for module in mc_test_module.modules()
    )
    assert not is_deterministic(mc_test_module, (10, 3, 10, 10))

    regular_module = mc_test_module.unpatch().eval()
    assert all(
        module.p != 0 for module in regular_module.modules() if isinstance(module, torch.nn.Dropout)
    )

    assert not any(
        isinstance(module, (WeightDropConv2d, WeightDropLinear))
        for module in regular_module.modules()
    )
    assert is_deterministic(regular_module, (10, 3, 10, 10))


if __name__ == '__main__':
    pytest.main()
