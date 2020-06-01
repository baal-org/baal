import sys
import copy
from abc import ABC, abstractmethod
from collections import OrderedDict

from collections.abc import Sequence

from typing import Dict, Any

import numpy as np
import structlog
import torch
from pydantic import BaseModel
from pytorch_lightning import LightningModule, Trainer, Callback
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from baal.active import ActiveLearningDataset, ActiveLearningLoop
from baal.active.heuristics import BALD
from baal.modelwrapper import mc_inference
from baal.utils.cuda_utils import to_cuda
from baal.utils.iterutils import map_on_tensor

log = structlog.get_logger('PL testing')


class ActiveLearningMixin(ABC):
    active_dataset = ...
    hparams = ...

    @abstractmethod
    def pool_loader(self):
        """DataLoader for the pool."""
        pass

    def predict_step(self, data, batch_idx):
        out = mc_inference(self, data, self.hparams.iterations, self.hparams.replicate_in_memory)
        return out

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = self.active_dataset.load_state_dict(checkpoint['active_dataset'])
        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['active_dataset'] = self.active_dataset.state_dict()


class ResetCallback(Callback):
    def __init__(self, weights):
        self.weights = weights

    def on_train_start(self, module):
        module.load(self.weights)


class BaalTrainer(Trainer):
    def predict_on_dataset(self, *args, **kwargs):
        preds = list(self.predict_on_dataset_generator())

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def predict_on_dataset_generator(self, *args, **kwargs):
        model = self.get_model()
        model.eval()
        dataloader = self.model.pool_loader()
        if len(dataloader) == 0:
            return None

        log.info("Start Predict", dataset=len(dataloader))
        for idx, (data, _) in enumerate(tqdm(dataloader, total=len(dataloader), file=sys.stdout)):
            if self.single_gpu:
                data = to_cuda(data)
            pred = self.model.predict_step(data, idx)
            yield map_on_tensor(lambda x: x.detach().cpu().numpy(), pred)


class VGG16(LightningModule, ActiveLearningMixin):
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
        self.vgg16 = vgg16(num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.vgg16(x)

    def log_hyperparams(self, *args):
        print(args)

    def save(self):
        print("SAVED")

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
            out = {key: torch.stack([x[key] for x in outputs]).mean() for key in outputs[0].keys()}
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


def main(hparams):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
    active_set = ActiveLearningDataset(
        CIFAR10(hparams.data_root, train=True, transform=train_transform, download=True),
        pool_specifics={
            'transform': test_transform
        })
    active_set.label_randomly(10)
    heuristic = BALD()
    model = VGG16(active_set, hparams)
    trainer = BaalTrainer(max_nb_epochs=3, default_save_path='/tmp',
                          callbacks=[ResetCallback(copy.deepcopy(model.state_dict()))])
    loop = ActiveLearningLoop(active_set, get_probabilities=trainer.predict_on_dataset_generator,
                              heuristic=heuristic,
                              ndata_to_label=hparams.query_size,
                              max_sample=hparams.max_sample)

    AL_STEPS = 100
    for al_step in range(AL_STEPS):
        trainer.fit(model)
        should_continue = loop.step()
        if not should_continue:
            break


if __name__ == '__main__':
    main(HParams())
