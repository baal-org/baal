import warnings

import pytest
import torch

from baal.bayesian.weight_drop import patch_module


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=10)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
        self.linear = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


@pytest.mark.parametrize("inplace", (True, False))
@pytest.mark.parametrize("layers", (['Linear'], ['Linear', 'Conv2d'], ['Conv2d']))
def test_patch_module_changes_weights(inplace, layers):
    test_module = SimpleModel()
    test_module.eval()
    simple_input = torch.randn(10, 3, 10, 10)
    assert torch.allclose(test_module(simple_input), test_module(simple_input))

    mc_test_module = patch_module(test_module, layers=layers, weight_dropout=0.2, inplace=inplace)

    # objects should be the same if inplace is True and not otherwise:
    assert (mc_test_module is test_module) == inplace
    assert not torch.allclose(mc_test_module(simple_input), mc_test_module(simple_input))

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


if __name__ == '__main__':
    pytest.main()
