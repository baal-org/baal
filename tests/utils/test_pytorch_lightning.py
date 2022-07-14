import copy
from dataclasses import dataclass, asdict

import numpy as np
import pytest
import torch
from baal.active import ActiveLearningDataset
from baal.utils.pytorch_lightning import ActiveLightningModule, BaalTrainer, ResetCallback, BaaLDataModule
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16


class SimpleDataset(Dataset):
    def __init__(self, mse=True):
        self.mse = mse

    def __len__(self):
        return 30

    def __getitem__(self, item):
        return torch.from_numpy(np.ones([3, 32, 32]) * item / 255.).float(), \
               (torch.FloatTensor([item % 2]))


@dataclass
class HParams:
    batch_size: int = 5
    data_root: str = '/tmp'
    num_classes: int = 10
    learning_rate: float = 0.001
    query_size: int = 5
    max_sample: int = -1
    iterations: int = 20
    replicate_in_memory: bool = True


class SimplePytorchLightning(ActiveLightningModule):
    def __init__(self, active_set, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.active_set = active_set
        self.vgg16 = vgg16()

    def forward(self, x):
        return self.vgg16(x)

    def pool_dataloader(self):
        return DataLoader(self.active_set.pool, batch_size=self.hparams.batch_size)

class MyDataModule(BaaLDataModule):
    pass


@pytest.fixture
def hparams():
    return asdict(HParams())


@pytest.fixture
def a_dataset():
    return SimpleDataset()


@pytest.fixture
def a_data_module(a_dataset, hparams):
    active_set = ActiveLearningDataset(a_dataset)
    active_set.label_randomly(10)
    return MyDataModule(active_dataset=active_set, batch_size=hparams['batch_size'])


@pytest.fixture
def a_pl_module(hparams, a_data_module):
    active_set = a_data_module.active_dataset
    return SimplePytorchLightning(active_set=active_set, **hparams)


def test_pool_dataloader(a_data_module):
    assert len(a_data_module.pool_dataloader()) == 4


def test_on_load_checkpoint(a_data_module, a_dataset, hparams):
    ckpt = {}
    _ = a_data_module.on_save_checkpoint(ckpt)
    assert ('active_dataset' in ckpt)
    active_set_2 = ActiveLearningDataset(a_dataset)
    data_mdoule_2 = MyDataModule(active_set_2, hparams['batch_size'])
    on_load_chkp = data_mdoule_2.on_load_checkpoint(ckpt)
    assert (len(a_data_module.active_dataset) == len(active_set_2))


def test_predict(a_data_module, a_pl_module):
    trainer = BaalTrainer(dataset=a_data_module.active_dataset,
                          max_epochs=3, default_root_dir='/tmp')
    active_set = a_data_module.active_dataset
    alt = trainer.predict_on_dataset(a_pl_module, a_data_module.pool_dataloader())
    assert len(alt) == len(active_set.pool)

    # Replicate = False works too!
    a_pl_module.hparams.replicate_in_memory = False
    alt = trainer.predict_on_dataset(a_pl_module, a_data_module.pool_dataloader())
    assert len(alt) == len(active_set.pool)


def test_reset_callback_resets_weights(a_data_module):
    def reset_fcs(model):
        """Reset all torch.nn.Linear layers."""

        def reset(m):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        model.apply(reset)

    model = vgg16()
    trainer = BaalTrainer(dataset=a_data_module.active_dataset,
                          max_epochs=3, default_root_dir='/tmp')
    trainer.fit_loop.current_epoch = 10
    initial_weights = copy.deepcopy(model.state_dict())
    initial_params = copy.deepcopy(list(model.parameters()))
    callback = ResetCallback(initial_weights)
    # Modify the params
    reset_fcs(model)
    new_params = model.parameters()
    assert not all(torch.eq(p1, p2).all() for p1, p2 in zip(initial_params, new_params))
    callback.on_train_start(trainer, model)
    new_params = model.parameters()
    assert all(torch.eq(p1, p2).all() for p1, p2 in zip(initial_params, new_params))
    assert trainer.current_epoch == 0


def test_pl_step(monkeypatch, a_data_module, a_pl_module, hparams):
    active_set = a_data_module.active_dataset
    trainer = BaalTrainer(dataset=active_set,
                          max_epochs=3, default_root_dir='/tmp',
                          query_size=hparams['query_size'])
    # Give everything.

    before = len(active_set)
    trainer.step(a_pl_module, a_data_module)
    after = len(active_set)

    assert after - before == hparams['query_size']
    # Add the lightning_module manually.
    trainer.strategy.connect(a_pl_module)

    # Only give the model
    before = len(active_set)
    trainer.step(a_pl_module)
    after = len(active_set)

    assert after - before == hparams['query_size']

    # No model, no dataloader
    before = len(active_set)
    trainer.step()
    after = len(active_set)

    assert after - before == hparams['query_size']