import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import structlog
import torch
from pydantic import BaseModel
from pytorch_lightning import LightningModule, Trainer
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

    def predict_step(self, batch, batch_idx):
        data, _ = batch
        out = mc_inference(self, data, self.hparams.iterations, self.hparams.replicate_in_memory)
        return out


class BaalTrainer(Trainer):
    def predict_on_dataset(self):
        preds = list(self.predict_on_dataset_generator())

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def predict_on_dataset_generator(self):
        self.model.eval()
        dataloader = self.model.pool_loader()
        if len(dataloader) == 0:
            return None

        log.info("Start Predict", dataset=len(dataloader))
        for idx, (data, _) in enumerate(tqdm(dataloader, total=len(dataloader), file=sys.stdout)):
            if self.single_gpu:
                data = to_cuda(data)
            pred = self.model.predict_on_batch(data, idx)
            yield map_on_tensor(lambda x: x.detach().cpu().numpy(), pred)


class VGG16(LightningModule, ActiveLearningMixin):
    def __init__(self, active_dataset, hparams):
        super().__init__()
        self.active_dataset = active_dataset
        self.hparams = hparams
        self.criterion = CrossEntropyLoss()
        self.model = vgg16(num_classes=hparams.num_classes)
        self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToTensor()])

    def forward(self, x):
        return self.model(x)

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
        loss_val = self.criterion(y, y_hat)

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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.active_dataset, self.hparams.batch_size, shuffle=True,
                          num_workers=4)

    def test_dataloader(self):
        ds = CIFAR10(root=self.hparams.data_root, train=False,
                     transform=self.test_transform, download=True)
        return DataLoader(ds, self.hparams.batch_size, shuffle=True,
                          num_workers=4)

    def pool_loader(self):
        return DataLoader(self.active_dataset.pool, self.hparams.batch_size, shuffle=False,
                          num_workers=4)


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
    heuristic = BALD()
    model = VGG16(active_set, hparams)
    trainer = BaalTrainer(model, max_nb_epochs=60)
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
