import sys
import types
import warnings
from collections.abc import Sequence
from typing import Dict, Any, Optional

import numpy as np
import structlog
import torch
from pytorch_lightning import Trainer, Callback, LightningDataModule, LightningModule
from pytorch_lightning.accelerators import GPUAccelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from baal.active import ActiveLearningDataset
from baal.active.heuristics import heuristics
from baal.modelwrapper import mc_inference
from baal.utils.cuda_utils import to_cuda
from baal.utils.iterutils import map_on_tensor

log = structlog.get_logger("PL testing")

warnings.warn(
    "baal.utils.pytorch_lightning is deprecated. BaaL is now integrated into Lightning Flash!"
    " Please see experiments/pytorch_lightning/lightning_flash_example.py for a new tutorial!",
    DeprecationWarning,
)


class BaaLDataModule(LightningDataModule):
    def __init__(self, active_dataset: ActiveLearningDataset, batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.active_dataset = active_dataset
        self.batch_size = batch_size

    def pool_dataloader(self) -> DataLoader:
        """Create Dataloader for the pool of unlabelled examples."""
        return DataLoader(
            self.active_dataset.pool, batch_size=self.batch_size, num_workers=4, shuffle=False
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "active_dataset" in checkpoint:
            self.active_dataset.load_state_dict(checkpoint["active_dataset"])
        else:
            log.warning("'active_dataset' not in checkpoint!")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["active_dataset"] = self.active_dataset.state_dict()


class ActiveLightningModule(LightningModule):
    """Pytorch Lightning class which adds methods to perform
    active learning.
    """

    def pool_dataloader(self) -> DataLoader:
        """DataLoader for the pool. Must be defined if you do not use a DataModule"""
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        """Predict on batch using MC inference `I` times.
        `I` is defined in the hparams property.
        Args:
            data (Tensor): Data to feed to the model.
            batch_idx (int): Batch index.
            dataloader_idx: Index of the current dataloader (not used)

        Returns:
            Models predictions stacked `I` times on the last axis.

        Notes:
            If `hparams.replicate_in_memeory` is True, we will stack inputs I times.
            This might create OoM errors. In that case, set it to False.
        """
        # Get the input only.
        x, _ = batch
        # Perform Monte-Carlo Inference fro I iterations.
        out = mc_inference(self, x, self.hparams.iterations, self.hparams.replicate_in_memory)
        return out


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
        trainer.fit_loop.current_epoch = 0


class BaalTrainer(Trainer):
    """Object that perform the training and active learning iteration.

    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        query_size (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        **kwargs: Parameters forwarded to `get_probabilities`
            and to pytorch_ligthning Trainer.__init__
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        query_size: int = 1,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.query_size = query_size
        self.heuristic = heuristic
        self.dataset = dataset
        self.kwargs = kwargs

    def predict_on_dataset(self, model=None, dataloader=None, *args, **kwargs):
        "For documentation, see `predict_on_dataset_generator`"
        preds = list(self.predict_on_dataset_generator(model, dataloader))

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def predict_on_dataset_generator(
        self, model=None, dataloader: Optional[DataLoader] = None, *args, **kwargs
    ):
        """Predict on the pool loader.

        Args:
            model: Model to be used in prediction. If None, will get the Trainer's model.
            dataloader (Optional[DataLoader]): If provided, will predict on this dataloader.
                                                Otherwise, uses model.pool_dataloader().

        Returns:
            Numpy arrays with all the predictions.
        """
        model = model or self.lightning_module
        model.eval()
        if isinstance(self.accelerator, GPUAccelerator):
            model.cuda(self.accelerator.root_device)
        dataloader = dataloader or model.pool_dataloader()
        if len(dataloader) == 0:
            return None

        log.info("Start Predict", dataset=len(dataloader))
        for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), file=sys.stdout)):
            if isinstance(self.accelerator, GPUAccelerator):
                batch = to_cuda(batch)
            pred = model.predict_step(batch, idx)
            yield map_on_tensor(lambda x: x.detach().cpu().numpy(), pred)
        # teardown, TODO customize this later?
        model.cpu()

    def step(self, model=None, datamodule: Optional[BaaLDataModule] = None) -> bool:
        """
        Perform an active learning step.

        model: Model to be used in prediction. If None, will get the Trainer's model.
        dataloader (Optional[DataLoader]): If provided, will predict on this dataloader.
                                                Otherwise, uses model.pool_dataloader().

        Notes:
            This will get the pool from the model pool_dataloader and if max_sample is set, it will
            **require** the data_loader sampler to select `max_pool` samples.

        Returns:
            boolean, Flag indicating if we continue training.

        """
        # High to low
        if datamodule is None:
            pool_dataloader = self.lightning_module.pool_dataloader()  # type: ignore
        else:
            pool_dataloader = datamodule.pool_dataloader()
        model = model if model is not None else self.lightning_module

        if isinstance(pool_dataloader.sampler, torch.utils.data.sampler.RandomSampler):
            log.warning(
                "Your pool_dataloader has `shuffle=True`," " it is best practice to turn this off."
            )

        if len(pool_dataloader) > 0:
            # TODO Add support for max_samples in pool_dataloader
            probs = self.predict_on_dataset_generator(
                model=model, dataloader=pool_dataloader, **self.kwargs
            )
            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                to_label = self.heuristic(probs)
                if len(to_label) > 0:
                    self.dataset.label(to_label[: self.query_size])
                    return True
        return False
