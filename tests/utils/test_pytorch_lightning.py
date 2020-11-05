import copy

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16

from baal.active import ActiveLearningDataset
from baal.utils.pytorch_lightning import ActiveLearningMixin, BaalTrainer, ResetCallback


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
    query_size: int = 10
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
    assert (len(model.pool_loader()) == 2)


def test_on_load_checkpoint():
    hparams = None
    dataset = DummyDataset()
    active_set = ActiveLearningDataset(dataset)
    active_set.label_randomly(10)
    model = DummyPytorchLightning(active_set, hparams)
    ckpt = {}
    save_chkp = model.on_save_checkpoint(ckpt)
    assert ('active_dataset' in ckpt)
    active_set_2 = ActiveLearningDataset(dataset)
    model_2 = DummyPytorchLightning(active_set_2, hparams)
    on_load_chkp = model_2.on_load_checkpoint(ckpt)
    assert (len(active_set) == len(active_set_2))


def test_predict():
    ckpt = {}
    hparams = HParams()
    dataset = DummyDataset()
    active_set = ActiveLearningDataset(dataset)
    active_set.label_randomly(10)
    model = DummyPytorchLightning(active_set, hparams)
    save_chkp = model.on_save_checkpoint(ckpt)
    trainer = BaalTrainer(dataset=active_set,
                          max_epochs=3, default_root_dir='/tmp',
                          callbacks=[ResetCallback(copy.deepcopy(save_chkp))])
    trainer.model = model
    alt = trainer.predict_on_dataset()
    assert len(alt) == len(active_set.pool)
    assert 'active_dataset' in save_chkp
    n_labelled = len(active_set)
    copy_save_chkp = copy.deepcopy(save_chkp)
    active_set.label_randomly(5)

    model.on_load_checkpoint(copy_save_chkp)
    assert len(active_set) == n_labelled


def test_reset_callback_resets_weights():
    def reset_fcs(model):
        """Reset all torch.nn.Linear layers."""

        def reset(m):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        model.apply(reset)

    model = vgg16()
    initial_weights = copy.deepcopy(model.state_dict())
    initial_params = copy.deepcopy(list(model.parameters()))
    callback = ResetCallback(initial_weights)
    # Modify the params
    reset_fcs(model)
    new_params = model.parameters()
    assert not all(torch.eq(p1, p2).all() for p1, p2 in zip(initial_params, new_params))
    callback.on_train_start(None, model)
    new_params = model.parameters()
    assert all(torch.eq(p1, p2).all() for p1, p2 in zip(initial_params, new_params))


def test_pl_step():
    hparams = HParams()
    dataset = DummyDataset()
    active_set = ActiveLearningDataset(dataset)
    active_set.label_randomly(10)
    model = DummyPytorchLightning(active_set, hparams)
    ckpt = {}
    save_chkp = model.on_save_checkpoint(ckpt)
    trainer = BaalTrainer(dataset=active_set,
                          max_epochs=3, default_root_dir='/tmp',
                          ndata_to_label=hparams.query_size,
                          callbacks=[ResetCallback(copy.deepcopy(save_chkp))])
    trainer.model = model

    before = len(active_set)
    trainer.step()
    after = len(active_set)

    assert after - before == hparams.query_size
