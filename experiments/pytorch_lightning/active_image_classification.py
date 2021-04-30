import copy
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import structlog
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16
from torchvision.transforms import transforms

from baal.active import ActiveLearningDataset, get_heuristic
from baal.bayesian.dropout import patch_module
from baal.utils.pytorch_lightning import ActiveLearningMixin, ResetCallback, BaalTrainer, BaaLDataModule

log = structlog.get_logger('PL testing')


class Cifar10DataModule(BaaLDataModule):
    def __init__(self, data_root, batch_size):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        active_set = ActiveLearningDataset(
            CIFAR10(data_root, train=True, transform=train_transform, download=True),
            pool_specifics={
                'transform': test_transform
            })
        self.test_set = CIFAR10(data_root, train=False, transform=test_transform, download=True)
        super().__init__(active_dataset=active_set, batch_size=batch_size,
                         train_transforms=train_transform,
                         test_transforms=test_transform)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.active_dataset, self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=True, num_workers=4)


class VGG16(LightningModule, ActiveLearningMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.name = "VGG16"
        self.version = "0.0.1"
        self.criterion = CrossEntropyLoss()
        self._build_model()

    def _build_model(self):
        # We use `patch_module` to swap Dropout modules in the model
        # for our implementation which enables MC-Dropou
        self.vgg16 = patch_module(vgg16(num_classes=self.hparams.num_classes))

    def forward(self, x):
        return self.vgg16(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch
        y_hat = self(x)

        # calculate loss
        loss_val = self.criterion(y_hat, y)

        self.log("train_loss", loss_val, prog_bar=True)
        return loss_val

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # calculate loss
        loss_val = self.criterion(y, y_hat)

        self.log("test_loss", loss_val, prog_bar=True)
        return loss_val

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer], []

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs)

    def epoch_end(self, outputs):
        out = {}
        if len(outputs) > 0:
            out = {key: torch.stack([x[key]
                                     for x in outputs]).mean()
                   for key in outputs[0].keys() if isinstance(key, torch.Tensor)}
        return out

    def test_epoch_end(self, outputs):
        return self.epoch_end(outputs)

    def training_epoch_end(self, outputs):
        return self.epoch_end(outputs)

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--iterations', type=int, default=20, help="Number of MC-Sampling to perform")
        parser.add_argument('--replicate_in_memory', type=bool, default=True)
        parser.add_argument('--batch_size', type=int, default=10)
        return parser


def parse_arguments():
    parser = ArgumentParser()
    parser = VGG16.add_model_specific_args(parser)
    parser = ArgumentParser(parents=[parser], conflict_handler='resolve', add_help=False)
    parser.add_argument('--heuristic', type=str, default='bald', help="Which heuristic to use.")
    parser.add_argument('--data_root', type=str, default='/tmp', help="Where to store data.")
    parser.add_argument('--query_size', type=int, default=100, help="How many items to label per step.")
    parser.add_argument('--gpus', type=int, default=1, help="How many GPUs to use.")
    return parser.parse_args()


def main():
    pl.seed_everything(42)
    args = parse_arguments()
    datamodule = Cifar10DataModule(args.data_root, batch_size=args.batch_size)
    datamodule.active_dataset.label_randomly(10)
    heuristic = get_heuristic(args.heuristic, shuffle_prop=0.0, reduction='none')
    model = VGG16(**vars(args))
    logger = TensorBoardLogger(save_dir=os.path.join('/tmp/', 'logs', 'active'), name='CIFAR10')
    trainer = BaalTrainer.from_argparse_args(args,
                                             # The weights of the model will change as it gets
                                             # trained; we need to keep a copy (deepcopy) so that
                                             # we can reset them.
                                             logger=logger,
                                             callbacks=[ResetCallback(copy.deepcopy(model.state_dict()))],
                                             dataset=datamodule.active_dataset,
                                             heuristic=heuristic,
                                             ndata_to_label=args.query_size
                                             )

    AL_STEPS = 100
    for al_step in range(AL_STEPS):
        print(f'Step {al_step} Dataset size {len(datamodule.active_dataset)}')
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
        should_continue = trainer.step(datamodule=datamodule)
        if not should_continue:
            break


if __name__ == '__main__':
    main()
