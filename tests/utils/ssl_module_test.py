import unittest
from argparse import Namespace

import torch
from baal.active import ActiveLearningDataset
from pytorch_lightning import Trainer
from src.baal.utils.ssl_module import SSLModule
from torch import nn
from torch.utils.data import ConcatDataset

from tests.utils.ssl_iterator_test import SSLTestDataset


class TestSSLModule(SSLModule):
    def __init__(self, active_dataset: ActiveLearningDataset, hparams: Namespace, **kwargs):
        super().__init__(active_dataset, hparams, **kwargs)
        self.linear = nn.Linear(784, 10)

        self.labeled_data = []
        self.unlabeled_data = []

    def forward(self, x, **kwargs):
        return self.linear(x)

    def supervised_training_step(self, batch, *args):
        self.labeled_data.extend(batch)
        return {'loss': torch.tensor(0., requires_grad=True)}

    def unsupervised_training_step(self, batch, *args):
        self.unlabeled_data.extend(batch)
        return {'loss': torch.tensor(0., requires_grad=True)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class SSLModuleTest(unittest.TestCase):
    def setUp(self):
        d1_len = 100
        d2_len = 1000
        d1 = SSLTestDataset(labeled=True, length=d1_len)
        d2 = SSLTestDataset(labeled=False, length=d2_len)
        dataset = ConcatDataset([d1, d2])

        print(len(dataset))

        self.al_dataset = ActiveLearningDataset(dataset)
        self.al_dataset.label(list(range(d1_len)))  # Label data from d1 (even numbers)

    def test_epoch(self):
        hparams = {
            'p': None,
            'num_steps': None,
            'batch_size': 10,
            'workers': 0}

        module = TestSSLModule(self.al_dataset, Namespace(**hparams))
        trainer = Trainer(max_epochs=1, num_sanity_val_steps=0, progress_bar_refresh_rate=0, logger=False,
                          checkpoint_callback=False)
        trainer.fit(module)

        assert len(module.labeled_data) == len(module.unlabeled_data)
        assert torch.all(torch.tensor(module.labeled_data) % 2 == 0)
        assert torch.all(torch.tensor(module.unlabeled_data) % 2 != 0)
