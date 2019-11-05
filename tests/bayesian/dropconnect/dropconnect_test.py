import pytest
import numpy as np
import torch
from torch.utils.data import Dataset

from baal.bayesian.dropconnect.weight_drop import WeightDropLinear, WeightDropConv2d,\
    patch_module, MCDropoutConnectModule


class DummyDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, item):
        return torch.from_numpy(np.ones([3, 10, 10]) * item / 255.).float(), torch.FloatTensor([item % 2])


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
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
@pytest.mark.parametrize("layers", (['Linear'], ['Linear', 'Conv2d']))
def test_patch_module_changes_weights(inplace, layers):

    test_module = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=10),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(8, 1),
    )

    conv_w = list(test_module.modules())[1].weight.clone().detach().numpy()
    linear_w = list(test_module.modules())[-1].weight.clone().detach().numpy()

    mc_test_module = patch_module(test_module, layers=layers, weight_dropout=0.2, inplace=inplace)

    # objects should be the same if inplace is True and not otherwise:
    assert (mc_test_module is test_module) == inplace

    new_linear_w = list(mc_test_module.modules())[-1].weight_raw.clone().detach().numpy()
    if layers == ['Linear']:
        assert isinstance(list(mc_test_module.modules())[-1], WeightDropLinear)
        assert isinstance(list(mc_test_module.modules())[1], torch.nn.Conv2d)
        new_conv_w = list(mc_test_module.modules())[1].weight.clone().detach().numpy()
        assert np.allclose(new_conv_w, conv_w)
        assert not np.allclose(new_linear_w, linear_w)
    else:
        assert isinstance(list(mc_test_module.modules())[-1], WeightDropLinear)
        assert isinstance(list(mc_test_module.modules())[1], WeightDropConv2d)
        new_conv_w = list(mc_test_module.modules())[1].weight_raw.clone().detach().numpy()
        assert not np.allclose(new_conv_w, conv_w)
        assert not np.allclose(new_linear_w, linear_w)

    assert list(mc_test_module.modules())[3].p == 0


def test_weight_change_after_forward_pass():
    test_module = DummyModel()
    dataset = DummyDataset()
    mc_test_module = MCDropoutConnectModule(test_module, layers=['Linear'], weight_dropout=0.2)

    assert not hasattr(list(test_module.modules())[-1], 'weight')
    linear_w = list(test_module.modules())[-1].weight_raw.clone().detach().numpy()

    input, _ = [torch.stack(v) for v in zip(*(dataset[0], dataset[1]))]
    mc_test_module.eval()
    out = mc_test_module(input)

    assert hasattr(list(test_module.modules())[-1], 'weight')
    new_linear_w = list(mc_test_module.modules())[-1].weight.clone().detach().numpy()
    assert not np.allclose(new_linear_w, linear_w)


if __name__ == '__main__':
    pytest.main()
