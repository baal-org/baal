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
        self.active_dataset.load_state_dict(checkpoint['active_dataset'])

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['active_dataset'] = self.active_dataset.state_dict()


class ResetCallback(Callback):
    def __init__(self, weights):
        self.weights = weights

    def on_train_start(self, trainer, module):
        module.load_state_dict(self.weights)


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
            if self.on_gpu:
                data = to_cuda(data)
            pred = self.model.predict_step(data, idx)
            yield map_on_tensor(lambda x: x.detach().cpu().numpy(), pred)
