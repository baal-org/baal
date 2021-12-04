import os

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms

from baal import ModelWrapper
from baal.active.heuristics import BALD
from baal.active.heuristics.heuristics_gpu import BALDGPUWrapper
from baal.bayesian import Dropout
from baal.bayesian.dropout import Dropout2d


class Flatten(nn.Module):
    def forward(self, x):
        return x.view([x.shape[0], -1])


class SimpleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 3, 32, 32)

    def __len__(self):
        return 100

    def __getitem__(self, item):
        return self.data[item], item % 10


@pytest.fixture
def classification_task(tmpdir):
    model = nn.Sequential(nn.Conv2d(3, 32, 3),
                          nn.ReLU(),
                          nn.Conv2d(32, 64, 3),
                          nn.MaxPool2d(2),
                          nn.AdaptiveAvgPool2d((7, 7)),
                          Flatten(),
                          nn.Linear(7 * 7 * 64, 128),
                          Dropout(),
                          nn.Linear(128, 10)
                          )
    model = ModelWrapper(model, nn.CrossEntropyLoss())
    test = SimpleDataset()
    return model, test


def test_bald_gpu(classification_task):
    torch.manual_seed(1337)
    model, test_set = classification_task
    wrap = BALDGPUWrapper(model, criterion=None)

    out = wrap.predict_on_dataset(test_set, 4, 10, False, 4)
    assert out.shape[0] == len(test_set)
    bald = BALD()
    torch.manual_seed(1337)
    out_bald = bald.get_uncertainties(model.predict_on_dataset(test_set, 4, 10, False, 4))
    assert np.allclose(out, out_bald, rtol=1e-5, atol=1e-5)


@pytest.fixture
def segmentation_task(tmpdir):
    model = nn.Sequential(nn.Conv2d(3, 32, 3),
                          nn.ReLU(),
                          nn.Conv2d(32, 64, 3),
                          nn.MaxPool2d(2),
                          nn.Conv2d(64, 64, 3),
                          Dropout2d(),
                          nn.ConvTranspose2d(64, 10, 3, 1)
                          )
    model = ModelWrapper(model, nn.CrossEntropyLoss())
    test = SimpleDataset()
    return model, test


def test_bald_gpu_seg(segmentation_task):
    torch.manual_seed(1337)
    model, test_set = segmentation_task
    wrap = BALDGPUWrapper(model, criterion=None, reduction='sum')

    out = wrap.predict_on_dataset(test_set, 4, 10, False, 4)
    assert out.shape[0] == len(test_set)
    bald = BALD(reduction='sum')
    torch.manual_seed(1337)
    out_bald = bald.get_uncertainties_generator(
        model.predict_on_dataset_generator(test_set, 4, 10, False, 4))
    assert np.allclose(out, out_bald, rtol=1e-5, atol=1e-5)
