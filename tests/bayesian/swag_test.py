import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.modules import Flatten
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from baal.bayesian.swag import StochasticWeightAveraging


def get_lr(swa):
    return swa.param_groups[0]['lr']


@pytest.fixture
def classification_dataset():
    class ClsDataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, item):
            return torch.randn(3, 32, 32) * item % 3, item % 3

    return ClsDataset()


@pytest.fixture
def base_optimizer():
    mod = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1),
                        nn.BatchNorm2d(8),
                        nn.Conv2d(8, 8, 3, padding=1),
                        nn.BatchNorm2d(8),
                        nn.AdaptiveAvgPool2d(1),
                        Flatten(),
                        nn.Linear(8, 2)
                        )
    mod.train()
    opt = SGD(mod.parameters(), lr=0.01)
    return opt, mod


def test_lr_is_cyclical(base_optimizer):
    base_optimizer, _ = base_optimizer
    SWA_FREQ = 100
    swa = StochasticWeightAveraging(base_optimizer, swa_start=10, swa_freq=SWA_FREQ,
                                    cycle_learning_rate=True,
                                    lr_max=0.01, lr_min=0.0001)
    acc = []
    for _ in range(10):
        swa.step()
        acc.append(get_lr(swa))
    assert all(lr == acc[0] for lr in acc)
    acc = []
    for _ in range(SWA_FREQ):
        swa.step()
        acc.append(get_lr(swa))
    assert all([acc[i] < acc[i - 1] for i in range(1, SWA_FREQ - 1)])
    assert acc[-1] == 0.01
    assert np.allclose(acc[-2], 0.0001, atol=1e-4)


def test_bn_updates(base_optimizer, classification_dataset):
    base_optimizer, model = base_optimizer

    def get_all_bn(model):
        return [module for module in model.modules() if
                not isinstance(module, nn.Sequential) and isinstance(module, nn.BatchNorm2d)]

    def get_means(bns):
        return [l.running_mean.numpy().copy() for l in bns]

    all_bns = get_all_bn(model)
    initial_means = get_means(all_bns)

    swa = StochasticWeightAveraging(base_optimizer, swa_start=0, swa_freq=100,
                                    cycle_learning_rate=True,
                                    lr_max=0.01, lr_min=0.0001)
    swa.bn_update(model, DataLoader(classification_dataset, batch_size=16))
    new_means = get_means(all_bns)

    assert not all(np.allclose(ini, new) for ini, new in zip(initial_means, new_means))


def test_sgd_is_deterministic(base_optimizer):
    base_optimizer, model = base_optimizer
    swa = StochasticWeightAveraging(base_optimizer, swa_start=0, swa_freq=10,
                                    cycle_learning_rate=True,
                                    lr_max=0.01, lr_min=0.0001)

    def compares_params(pr1, pr2):
        return all(torch.eq(p1, p2).all() for p1, p2 in zip(pr1, pr2))

    get_params = lambda model: list(map(lambda x: x.clone(), model.parameters()))
    init_params = get_params(model)
    for _ in range(10):
        swa.step()
    for _ in range(2):
        swa.sgd()
        assert compares_params(init_params, get_params(model))
    for _ in range(2):
        swa.swa()
        assert compares_params(init_params, get_params(model))


def test_sample_is_stochastic(base_optimizer):
    base_optimizer, model = base_optimizer
    swa = StochasticWeightAveraging(base_optimizer, swa_start=0, swa_freq=10,
                                    cycle_learning_rate=True,
                                    lr_max=0.01, lr_min=0.0001)

    def compares_params(pr1, pr2):
        return all(torch.eq(p1, p2).all() for p1, p2 in zip(pr1, pr2))

    get_params = lambda model: list(map(lambda x: x.clone(), model.parameters()))
    init_params = get_params(model)
    with pytest.raises(ValueError):
        # We did not train yet!
        swa.sample()
    for _ in range(10):
        swa.step()
    for _ in range(2):
        swa.sample()
        assert not compares_params(init_params, get_params(model))


if __name__ == '__main__':
    pytest.main()
