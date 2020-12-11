"""
Semi-supervised model for classification.
Pi-Model from TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED LEARNING (Laine 2017).
https://arxiv.org/abs/1610.02242
"""

import argparse
from argparse import Namespace
from typing import Dict

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.hub import load_state_dict_from_url

from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import vgg11

from baal.active import ActiveLearningDataset
from baal.utils.ssl_module import SSLModule

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class GaussianNoise(nn.Module):
    """ Add random gaussian noise to images."""

    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn(x.size()).type_as(x) * self.std


class RandomTranslation(nn.Module):
    """Randomly translate images."""

    def __init__(self, augment_translation=10):
        super(RandomTranslation, self).__init__()
        self.augment_translation = augment_translation

    def forward(self, x):
        """
            Randomly translate images.
        Args:
            x (Tensor) : (N, C, H, W) image tensor

        Returns:
            (N, C, H, W) translated image tensor
        """
        batch_size = len(x)

        t_min = -self.augment_translation / x.shape[-1]
        t_max = (self.augment_translation + 1) / x.shape[-1]

        matrix = torch.eye(3)[None].repeat((batch_size, 1, 1))
        tx = (t_min - t_max) * torch.rand(batch_size) + t_max
        ty = (t_min - t_max) * torch.rand(batch_size) + t_max

        matrix[:, 0, 2] = tx
        matrix[:, 1, 2] = ty
        matrix = matrix[:, 0:2, :]

        grid = nn.functional.affine_grid(matrix, x.shape).type_as(x)
        x = nn.functional.grid_sample(x, grid)

        return x


class PIModel(SSLModule):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize(3 * [0.5], 3 * [0.5])])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(3 * [0.5], 3 * [0.5])])

    def __init__(self, active_dataset: ActiveLearningDataset, hparams: Namespace,
                 network: nn.Module):
        super().__init__(active_dataset, hparams)

        self.network = network

        # Maximum unsupervised loss weight as defined in the paper.
        M = len(self.active_dataset)
        N = (len(self.active_dataset) + len(self.active_dataset.pool))
        self.max_unsupervised_weight = self.hparams.w_max * M / N

        self.criterion = nn.CrossEntropyLoss()
        self.consistency_criterion = nn.MSELoss()

        if self.hparams.baseline:
            assert self.hparams.p == 1, "Only labeled data is used for baseline (p=1)"

        # Consistency augmentations
        self.gaussian_noise = GaussianNoise()
        self.random_crop = RandomTranslation()

        # Keep track of current lr for logging
        self.current_lr = self.hparams.lr

    def forward(self, x):
        if self.training and not self.hparams.no_augmentations:
            x = self.random_crop(x)
            x = self.gaussian_noise(x)

        return self.network(x)

    def supervised_training_step(self, batch, *args) -> Dict:
        x, y = batch

        z = self.forward(x)

        supervised_loss = self.criterion(z, y)

        accuracy = (y == z.argmax(-1)).float().sum() / len(x)

        logs = {'criterion_loss': supervised_loss, 'accuracy': accuracy}

        if not self.hparams.baseline:
            with torch.no_grad():
                z_hat = self.forward(x)

            unsupervised_loss = self.consistency_criterion(z, z_hat)
            unsupervised_weight = self.max_unsupervised_weight * self.rampup_value()
            loss = supervised_loss + unsupervised_weight * unsupervised_loss

            logs.update({'supervised_consistency_loss': unsupervised_loss,
                         'unsupervised_weight': unsupervised_weight})

        else:
            loss = supervised_loss

        logs.update({'supervised_loss': loss,
                     'rampup_value': self.rampup_value(),
                     'learning_rate': self.current_lr
                     })

        return {'loss': loss, 'log': logs}

    def unsupervised_training_step(self, batch, *args) -> Dict:
        x, _ = batch

        with torch.no_grad():
            z = self.forward(x)
        z_hat = self.forward(x)

        unsupervised_loss = self.consistency_criterion(z, z_hat)
        unsupervised_weight = self.max_unsupervised_weight * self.rampup_value()
        loss = unsupervised_weight * unsupervised_loss

        logs = {'unsupervised_consistency_loss': unsupervised_loss,
                'unsupervised_loss': loss}

        return {'loss': loss, 'log': logs}

    def rampup_value(self):
        if self.current_epoch <= self.hparams.rampup_stop - 1:
            T = (1 / (self.hparams.rampup_stop - 1)) * self.current_epoch
            return np.exp(-5 * (1 - T) ** 2)
        else:
            return 1

    def rampdown_value(self):
        if self.current_epoch >= self.epoch - self.hparams.rampup_stop - 1:
            T = (1 / (self.epoch - self.hparams.rampup_stop - 1)) * self.current_epoch
            return np.exp(-12.5 * T ** 2)
        else:
            return 0

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9,
                               weight_decay=1e-4)

    def test_val_step(self, batch: int, prefix: str) -> Dict[str, Tensor]:
        x, y = batch
        y_hat = self(x)

        loss_val = self.criterion(y_hat, y)
        accuracy = (y == y_hat.argmax(-1)).float().sum() / len(x)

        output = {'{}_loss'.format(prefix): loss_val, '{}_accuracy'.format(prefix): accuracy}

        return output

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        return self.test_val_step(batch, prefix='val')

    def test_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        return self.test_val_step(batch, prefix='test')

    def val_dataloader(self):
        ds = CIFAR10(root=self.hparams.data_root, train=False, transform=self.test_transform,
                     download=True)
        return DataLoader(ds, self.hparams.batch_size, shuffle=False, num_workers=self.hparams.workers)

    def test_dataloader(self):
        ds = CIFAR10(root=self.hparams.data_root, train=False, transform=self.test_transform,
                     download=True)
        return DataLoader(ds, self.hparams.batch_size, shuffle=False, num_workers=self.hparams.workers)

    def epoch_end(self, outputs):
        avg_metrics = {}
        for key in outputs[0].keys():
            if isinstance(outputs[0][key], torch.Tensor):
                avg_metrics[key] = torch.stack([x[key] for x in outputs]).mean()

        output = {}
        output['progress_bar'] = avg_metrics
        output['log'] = avg_metrics
        output['log']['step'] = self.current_epoch

        return output

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs)

    def test_epoch_end(self, outputs):
        return self.epoch_end(outputs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific arguments to argparser.

        Args:
            parent_parser (argparse.ArgumentParser): parent parser to which to add arguments

        Returns:
            argparser with added arguments
        """
        parser = super(PIModel, PIModel).add_model_specific_args(parent_parser)
        parser.add_argument('--baseline', action='store_true')
        parser.add_argument('--rampup_stop', default=80)
        parser.add_argument('--epochs', default=300, type=int)
        parser.add_argument('--batch-size', default=100, type=int, help='batch size')
        parser.add_argument('--lr', default=0.003, type=float, help='Max learning rate', dest='lr')
        parser.add_argument('--w_max', default=100, type=float,
                            help='Maximum unsupervised weight, default=100 for CIFAR10 as '
                                 'described in paper')
        parser.add_argument('--no_augmentations', action='store_true')
        return parser


if __name__ == '__main__':
    from pytorch_lightning import Trainer, seed_everything
    from argparse import ArgumentParser

    args = ArgumentParser(add_help=False)
    args.add_argument('--data-root', default='./', type=str, help='Where to download the data')
    args.add_argument('--gpus', default=torch.cuda.device_count(), type=int)
    args.add_argument('--num_labeled', default=5000, type=int)
    args.add_argument('--seed', default=None, type=int)
    args = PIModel.add_model_specific_args(args)
    params = args.parse_args()

    seed = seed_everything(params.seed)

    active_set = ActiveLearningDataset(
        CIFAR10(params.data_root, train=True, transform=PIModel.train_transform, download=True),
        pool_specifics={'transform': PIModel.test_transform})
    active_set.label_randomly(params.num_labeled)

    print("Active set length: {}".format(len(active_set)))
    print("Pool set length: {}".format(len(active_set.pool)))

    net = vgg11(pretrained=False, num_classes=10)

    weights = load_state_dict_from_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth')
    weights = {k: v for k, v in weights.items() if 'classifier.6' not in k}
    net.load_state_dict(weights, strict=False)

    system = PIModel(network=net, active_dataset=active_set, hparams=params)

    trainer = Trainer(num_sanity_val_steps=0, max_epochs=params.epochs,
                      early_stop_callback=False, gpus=params.gpus)

    trainer.fit(system)

    trainer.test(ckpt_path='best')
