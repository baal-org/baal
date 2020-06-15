import pytest
import numpy as np
import torch
from PIL import Image
from hypothesis.extra import numpy as np_strategies
from hypothesis import given

from baal.utils.pytorch_lightning import ActiveLearningMixin
from baal.active import ActiveLearningDataset

from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, mse=True):
        self.mse = mse

    def __len__(self):
        return 20

    def __getitem__(self, item):
        return torch.from_numpy(np.ones([3, 10, 10]) * item / 255.).float(), \
                               (torch.FloatTensor([item % 2]))


class DummyPytorchLightning(ActiveLearningMixin):
    def pool_loader(self):
        dataset = DummyDataset()
        return DataLoader(dataset, 5, shuffle=False, num_workers=4)


def test_active_learning_mixin():
    dataset = DummyDataset()
    active_set = ActiveLearningDataset(dataset)
    active_set.label_randomly(10)
    model = DummyPytorchLightning()
    assert(len(model.pool_loader()) == 4)
