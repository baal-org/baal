import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Dict, Any

import numpy as np
import structlog
from baal.active import ActiveLearningDataset
from baal.active.heuristics import heuristics
from baal.modelwrapper import mc_inference
from baal.utils.cuda_utils import to_cuda
from baal.utils.iterutils import map_on_tensor
from pytorch_lightning import Trainer, Callback
from tqdm import tqdm

log = structlog.get_logger('PL testing')


class ActiveLearningMixin(ABC):
    """Pytorch Lightning Mixin which adds methods to perform
    active learning.
    """
    active_dataset = ...
    hparams = ...

    @abstractmethod
    def pool_loader(self):
        """DataLoader for the pool."""
        pass

    def predict_step(self, data, batch_idx):
        """Predict on batch using MC inference `I` times.
        `I` is defined in the hparams property.
        Args:
            data (Tensor): Data to feed to the model.
            batch_idx (int): Batch index.

        Returns:
            Models predictions stacked `I` times on the last axis.
        """
        out = mc_inference(self, data, self.hparams.iterations, self.hparams.replicate_in_memory)
        return out

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.active_dataset.load_state_dict(checkpoint['active_dataset'])

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['active_dataset'] = self.active_dataset.state_dict()
        return checkpoint


class ResetCallback(Callback):
    """Callback to reset the weights between active learning steps.

    Args:
        weights (dict): State dict of the model.

    Notes:
        The weight should be deep copied beforehand.

    """
    def __init__(self, weights):
        self.weights = weights

    def on_train_start(self, trainer, module):
        """Will reset the module to its initial weights."""
        module.load_state_dict(self.weights)


class BaalTrainer(Trainer):
    """Object that perform the training and active learning iteration.

    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        ndata_to_label (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        **kwargs: Parameters forwarded to `get_probabilities`
            and to pytorch_ligthning Trainer.__init__
    """

    def __init__(self, dataset: ActiveLearningDataset,
                 heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
                 ndata_to_label: int = 1,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.ndata_to_label = ndata_to_label
        self.heuristic = heuristic
        self.dataset = dataset
        self.kwargs = kwargs

    def predict_on_dataset(self, *args, **kwargs):
        """Predict on the pool loader.

        Returns:
            Numpy arrays with all the predictions.
        """
        preds = list(self.predict_on_dataset_generator())

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def predict_on_dataset_generator(self, *args, **kwargs):
        """Predict on the pool loader.

        Returns:
            Numpy arrays with all the predictions.
        """
        model = self.get_model()
        model.eval()
        if self.on_gpu:
            model.cuda(self.root_gpu)
        dataloader = self.model.pool_loader()
        if len(dataloader) == 0:
            return None

        log.info("Start Predict", dataset=len(dataloader))
        for idx, (data, _) in enumerate(tqdm(dataloader, total=len(dataloader), file=sys.stdout)):
            if self.on_gpu:
                data = to_cuda(data)
            pred = self.model.predict_step(data, idx)
            yield map_on_tensor(lambda x: x.detach().cpu().numpy(), pred)
        # teardown, TODO customize this later?
        model.cpu()

    def step(self) -> bool:
        """
        Perform an active learning step.

        Returns:
            boolean, Flag indicating if we continue training.

        """

        indices = None  # TODO Add support for max_samples in pool_loader

        if len(self.get_model().active_dataset.pool) > 0:
            probs = self.predict_on_dataset_generator(**self.kwargs)
            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                to_label = self.heuristic(probs)
                if indices is not None:
                    to_label = indices[np.array(to_label)]
                if len(to_label) > 0:
                    self.dataset.label(to_label[: self.ndata_to_label])
                    return True
        return False
