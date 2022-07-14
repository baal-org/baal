import argparse
import typing
from argparse import Namespace
from typing import Dict

import pytorch_lightning as pl

from baal.active import ActiveLearningDataset
from baal.utils.ssl_iterator import SemiSupervisedIterator


class SSLModule(pl.LightningModule):
    """
        Pytorch Lightning module for semi-supervised learning.

    Args:
        active_dataset (ActiveLearningDataset): active learning dataset
        hparams (Namespace): hyper-parameters for the module
        **kwargs (**dict): extra arguments
    """

    def __init__(self, active_dataset: ActiveLearningDataset, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.active_dataset = active_dataset

    def supervised_training_step(self, batch, *args) -> Dict:
        raise NotImplementedError

    def unsupervised_training_step(self, batch, *args) -> Dict:
        raise NotImplementedError

    def training_step(self, batch, *args):
        if SemiSupervisedIterator.is_labeled(batch):
            return self.supervised_training_step(SemiSupervisedIterator.get_batch(batch), *args)
        else:
            return self.unsupervised_training_step(SemiSupervisedIterator.get_batch(batch), *args)

    @typing.no_type_check
    def train_dataloader(self) -> SemiSupervisedIterator:
        """SemiSupervisedIterator for train set.

        Returns:
            SemiSupervisedIterator on the train set + pool set.
        """
        return SemiSupervisedIterator(
            self.active_dataset,
            self.hparams.batch_size,
            num_steps=self.hparams.num_steps,
            p=self.hparams.p,
            num_workers=self.hparams.workers,
            shuffle=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific arguments to argparser.

        Args:
            parent_parser (argparse.ArgumentParser): parent parser to which to add arguments

        Returns:
            argparser with added arguments
        """
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, conflict_handler="resolve"
        )
        parser.add_argument(
            "--p", default=None, type=float, help="Probability of selecting labeled batch"
        )
        parser.add_argument("--num_steps", default=None, type=int, help="Number of steps per epoch")
        parser.add_argument("--batch-size", default=32, type=int)
        parser.add_argument("--workers", default=4, type=int)

        return parser
