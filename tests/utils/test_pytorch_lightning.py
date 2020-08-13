import unittest
import pytest
import numpy as np
import torch
import copy

from collections import OrderedDict

from torch import nn
from PIL import Image
from hypothesis.extra import numpy as np_strategies
from hypothesis import given

from baal.utils.pytorch_lightning import ActiveLearningMixin, BaalTrainer, ResetCallback
from baal.active import ActiveLearningDataset

from baal.modelwrapper import ModelWrapper

from torch import optim
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule
from torchvision.models import vgg16


class DummyDataset(Dataset):
    def __init__(self, mse=True):
        self.mse = mse

    def __len__(self):
        return 20

    def __getitem__(self, item):
        return torch.from_numpy(np.ones([3, 32, 32]) * item / 255.).float(), \
                               (torch.FloatTensor([item % 2]))


class HParams():
    batch_size: int = 10
    data_root: str = '/tmp'
    num_classes: int = 10
    learning_rate: float = 0.001
    query_size: int = 100
    max_sample: int = -1
    iterations: int = 20
    replicate_in_memory: bool = True


class DummyPytorchLightning(ActiveLearningMixin, LightningModule):
    def __init__(self, dataset, hparams):
        super().__init__()
        self.active_dataset = dataset
        self.hparams = hparams
        self.vgg16 = vgg16()

    def forward(self, x):
        return self.vgg16(x)

    def pool_loader(self):
        return DataLoader(self.active_dataset.pool, 5, shuffle=False, num_workers=4)


def test_active_learning_mixin():
    hparams = None
    dataset = DummyDataset()
    active_set = ActiveLearningDataset(dataset)
    active_set.label_randomly(10)
    model = DummyPytorchLightning(active_set, hparams)
    assert(len(model.pool_loader()) == 2)


def test_on_load_checkpoint():
    hparams = None
    dataset = DummyDataset()
    active_set = ActiveLearningDataset(dataset)
    active_set.label_randomly(10)
    model = DummyPytorchLightning(active_set, hparams)
    ckpt = {}
    save_chkp = model.on_save_checkpoint(ckpt)
    assert('active_dataset' in ckpt)
    active_set_2 = ActiveLearningDataset(dataset)
    model_2 = DummyPytorchLightning(active_set_2, hparams)
    on_load_chkp = model_2.on_load_checkpoint(ckpt)
    assert(len(active_set) == len(active_set_2))


def test_predict():
    ckpt = {}
    hparams = HParams()
    dataset = DummyDataset()
    active_set = ActiveLearningDataset(dataset)
    active_set.label_randomly(10)
    model = DummyPytorchLightning(active_set, hparams)
    save_chkp = model.on_save_checkpoint(ckpt)
    trainer = BaalTrainer(max_nb_epochs=3, default_save_path='/tmp',
                          callbacks=[ResetCallback(copy.deepcopy(save_chkp))])
    trainer.model = model
    alt = trainer.predict_on_dataset()
    assert len(alt) == len(active_set.pool)
