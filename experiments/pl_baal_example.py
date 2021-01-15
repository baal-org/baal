import copy
from collections import OrderedDict

import structlog
import torch
from baal.active import ActiveLearningDataset
from baal.active.heuristics import BALD
from baal.bayesian.dropout import patch_module
from baal.utils.pytorch_lightning import ActiveLearningMixin, ResetCallback, BaalTrainer
from pytorch_lightning import LightningModule
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16
from torchvision.transforms import transforms

log = structlog.get_logger('PL testing')

try:
    from pydantic import BaseModel
except ImportError:
    raise ValueError('pydantic is required for this example.\n pip install pydantic')


class VGG16(ActiveLearningMixin, LightningModule):
    def __init__(self, active_dataset, hparams):
        super().__init__()
        self.name = "VGG16"
        self.version = "0.0.1"
        self.active_dataset = active_dataset
        self.hparams = hparams
        self.criterion = CrossEntropyLoss()

        self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToTensor()])
        self._build_model()

    def _build_model(self):
        # We use `patch_module` to swap Dropout modules in the model
        # for our implementation which enables MC-Dropou
        self.vgg16 = patch_module(vgg16(num_classes=self.hparams.num_classes))

    def forward(self, x):
        return self.vgg16(x)

    def log_hyperparams(self, *args):
        print(args)

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

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # calculate loss
        loss_val = self.criterion(y, y_hat)

        tqdm_dict = {'val_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer], []

    def train_dataloader(self):
        return DataLoader(self.active_dataset, self.hparams.batch_size, shuffle=True,
                          num_workers=4)

    def test_dataloader(self):
        ds = CIFAR10(root=self.hparams.data_root, train=False,
                     transform=self.test_transform, download=True)
        return DataLoader(ds, self.hparams.batch_size, shuffle=False,
                          num_workers=4)

    def pool_loader(self):
        return DataLoader(self.active_dataset.pool, self.hparams.batch_size, shuffle=False,
                          num_workers=4)

    def log_metrics(self, metrics, step_num):
        print('Epoch', step_num, metrics)

    def agg_and_log_metrics(self, metrics, step):
        self.log_metrics(metrics, step)

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


class HParams(BaseModel):
    batch_size: int = 10
    data_root: str = '/tmp'
    num_classes: int = 10
    learning_rate: float = 0.001
    query_size: int = 100
    max_sample: int = -1
    iterations: int = 20
    replicate_in_memory: bool = True
    n_gpus: int = torch.cuda.device_count()


def main(hparams):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    active_set = ActiveLearningDataset(CIFAR10(hparams.data_root, train=True, transform=train_transform, download=True),
                                       pool_specifics={
                                           'transform': test_transform
                                       })
    active_set.label_randomly(10)
    heuristic = BALD()
    model = VGG16(active_set, hparams)
    dp = 'dp' if hparams.n_gpus > 1 else None
    trainer = BaalTrainer(max_epochs=3, default_root_dir=hparams.data_root,
                          gpus=hparams.n_gpus, distributed_backend=dp,
                          # The weights of the model will change as it gets
                          # trained; we need to keep a copy (deepcopy) so that
                          # we can reset them.
                          callbacks=[ResetCallback(copy.deepcopy(model.state_dict()))],
                          dataset=active_set,
                          heuristic=heuristic,
                          ndata_to_label=hparams.query_size
                          )

    AL_STEPS = 100
    for al_step in range(AL_STEPS):
        # TODO Issue 95 Make PL trainer epoch self-aware
        trainer.current_epoch = 0
        print(f'Step {al_step} Dataset size {len(active_set)}')
        trainer.fit(model)
        should_continue = trainer.step()
        if not should_continue:
            break


if __name__ == '__main__':
    main(HParams())
