import sys
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Union, List

import numpy as np
import structlog
import torch
from numpy._typing import NDArray
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm.autonotebook import tqdm

from baal.active.dataset.base import Dataset
from baal.metrics.mixin import MetricMixin
from baal.utils.array_utils import stack_in_memory
from baal.utils.cuda_utils import to_cuda
from baal.utils.equality import assert_not_none
from baal.utils.iterutils import map_on_tensor
from baal.utils.metrics import Loss
from baal.utils.warnings import raise_warnings_cache_replicated

log = structlog.get_logger("baal")


def _stack_preds(out):
    if isinstance(out[0], Sequence):
        out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
    else:
        out = torch.stack(out, dim=-1)
    return out


@dataclass
class TrainingArgs:
    optimizer: Optional[Optimizer] = None
    batch_size: int = 32
    epoch: int = 0
    use_cuda: bool = torch.cuda.is_available()
    workers: int = 4
    collate_fn: Callable = default_collate
    regularizer: Optional[Callable] = None
    criterion: Optional[Callable] = None
    replicate_in_memory: bool = True


class ModelWrapper(MetricMixin):
    """
    Wrapper created to ease the training/testing/loading.

    Args:
        model (nn.Module): The model to optimize.
        args (TrainingArgs): Model arguments for training/predicting.
    """

    def __init__(self, model, args: TrainingArgs):
        self.model = model
        self.args = args
        self.metrics = dict()
        self.active_learning_metrics = defaultdict(dict)
        self.add_metric("loss", lambda: Loss())
        self._active_dataset_size = -1

        raise_warnings_cache_replicated(
            self.model, replicate_in_memory=self.args.replicate_in_memory
        )

    def train_on_dataset(self, dataset):
        """
        Train for `epoch` epochs on a Dataset `dataset.

        Args:
            dataset (Dataset): Pytorch Dataset to be trained on.

        Returns:
            The training history.
        """
        dataset_size = len(dataset)
        self.train()
        self.set_dataset_size(dataset_size)
        history = []
        log.info("Starting training", epoch=self.args.epoch, dataset=dataset_size)
        for _ in range(self.args.epoch):
            self._reset_metrics("train")
            for data, target, *_ in DataLoader(
                dataset,
                self.args.batch_size,
                True,
                num_workers=self.args.workers,
                collate_fn=self.args.collate_fn,
            ):
                _ = self.train_on_batch(data, target)
            history.append(self.get_metrics("train")["train_loss"])

        self.args.optimizer.zero_grad()  # Assert that the gradient is flushed.
        log.info("Training complete", train_loss=self.get_metrics("train")["train_loss"])
        self.active_step(dataset_size, self.get_metrics("train"))
        return history

    def test_on_dataset(
        self,
        dataset: Dataset,
        average_predictions: int = 1,
    ):
        """
        Test the model on a Dataset `dataset`.

        Args:
            dataset (Dataset): Dataset to evaluate on.
            average_predictions (int): The number of predictions to average to
                compute the test loss.

        Returns:
            Average loss value over the dataset.
        """
        self.eval()
        log.info("Starting evaluating", dataset=len(dataset))
        self._reset_metrics("test")

        for data, target, *_ in DataLoader(
            dataset,
            self.args.batch_size,
            False,
            num_workers=self.args.workers,
            collate_fn=self.args.collate_fn,
        ):
            _ = self.test_on_batch(data, target, average_predictions=average_predictions)

        log.info("Evaluation complete", test_loss=self.get_metrics("test")["test_loss"])
        self.active_step(None, self.get_metrics("test"))
        return self.get_metrics("test")["test_loss"]

    def train_and_test_on_datasets(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        return_best_weights=False,
        patience=None,
        min_epoch_for_es=0,
        skip_epochs=1,
    ):
        """
        Train and test the model on both Dataset `train_dataset`, `test_dataset`.

        Args:
            train_dataset (Dataset): Dataset to train on.
            test_dataset (Dataset): Dataset to evaluate on.
            return_best_weights (bool): If True, will keep the best weights and return them.
            patience (Optional[int]): If provided, will use early stopping to stop after
                                        `patience` epoch without improvement.
            min_epoch_for_es (int): Epoch at which the early stopping starts.
            skip_epochs (int): Number of epochs to skip for test_on_dataset

        Returns:
            History and best weights if required.
        """
        best_weight = None
        best_loss = 1e10
        best_epoch = 0
        hist = []
        for e in range(self.args.epoch):
            _ = self.train_on_dataset(
                train_dataset,
            )
            if e % skip_epochs == 0:
                te_loss = self.test_on_dataset(test_dataset)
                hist.append(self.get_metrics())
                if te_loss < best_loss:
                    best_epoch = e
                    best_loss = te_loss
                    if return_best_weights:
                        best_weight = deepcopy(self.state_dict())

                if patience is not None and (e - best_epoch) > patience and (e > min_epoch_for_es):
                    # Early stopping
                    break
            else:
                hist.append(self.get_metrics("train"))

        if return_best_weights:
            return hist, best_weight
        else:
            return hist

    def predict_on_dataset_generator(
        self,
        dataset: Dataset,
        iterations: int,
        half=False,
        verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time.

        Args:
            dataset (Dataset): Dataset to predict on.
            iterations (int): Number of iterations per sample.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to display progress

        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Generators [batch_size, n_classes, ..., n_iterations].
        """
        self.eval()
        if len(dataset) == 0:
            return None

        log.info("Start Predict", dataset=len(dataset))
        loader = DataLoader(
            dataset,
            self.args.batch_size,
            False,
            num_workers=self.args.workers,
            collate_fn=self.args.collate_fn,
        )
        if verbose:
            loader = tqdm(loader, total=len(loader), file=sys.stdout)
        for idx, (data, *_) in enumerate(loader):

            pred = self.predict_on_batch(data, iterations)
            pred = map_on_tensor(lambda x: x.detach(), pred)
            if half:
                pred = map_on_tensor(lambda x: x.half(), pred)
            yield map_on_tensor(lambda x: x.cpu().numpy(), pred)

    def predict_on_dataset(
        self,
        dataset: Dataset,
        iterations: int,
        half=False,
        verbose=True,
    ) -> Union[NDArray, List[NDArray]]:
        """
        Use the model to predict on a dataset `iterations` time.

        Args:
            dataset (Dataset): Dataset to predict on.
            iterations (int): Number of iterations per sample.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to show progress.

        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Array [n_samples, n_outputs, ..., n_iterations].
        """
        preds = list(
            self.predict_on_dataset_generator(
                dataset=dataset,
                iterations=iterations,
                half=half,
                verbose=verbose,
            )
        )

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def train_on_batch(self, data, target):
        """
        Train the current model on a batch using `optimizer`.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.

        Returns:
            Tensor, the loss computed from the criterion.
        """

        if self.args.use_cuda:
            data, target = to_cuda(data), to_cuda(target)
        self.args.optimizer.zero_grad()
        output = self.model(data)
        loss = self.args.criterion(output, target)

        if self.args.regularizer:
            regularized_loss = loss + self.args.regularizer()
            regularized_loss.backward()
        else:
            loss.backward()

        self.args.optimizer.step()
        self._update_metrics(output, target, loss, filter="train")
        return loss

    def test_on_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        average_predictions: int = 1,
    ):
        """
        Test the current model on a batch.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            average_predictions (int): The number of predictions to average to
                compute the test loss.

        Returns:
            Tensor, the loss computed from the criterion.
        """
        with torch.no_grad():
            if self.args.use_cuda:
                data, target = to_cuda(data), to_cuda(target)

            preds = map_on_tensor(
                lambda p: p.mean(-1),
                self.predict_on_batch(data, iterations=average_predictions),
            )
            loss = assert_not_none(self.args.criterion)(preds, target)
            self._update_metrics(preds, target, loss, "test")
            return loss

    def predict_on_batch(self, data, iterations=1):
        """
        Get the model's prediction on a batch.

        Args:
            data (Tensor): The model input.
            iterations (int): Number of prediction to perform.

        Returns:
            Tensor, the loss computed from the criterion.
                    shape = {batch_size, nclass, n_iteration}.

        Raises:
            Raises RuntimeError if CUDA rans out of memory during data replication.
        """
        with torch.no_grad():
            if self.args.use_cuda:
                data = to_cuda(data)
            if self.args.replicate_in_memory:
                data = map_on_tensor(lambda d: stack_in_memory(d, iterations), data)
                try:
                    out = self.model(data)
                except RuntimeError as e:
                    raise RuntimeError(
                        """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
                    Use `replicate_in_memory=False` in order to reduce the memory requirements.
                    Note that there will be some speed trade-offs"""
                    ) from e
                out = map_on_tensor(lambda o: o.view([iterations, -1, *o.size()[1:]]), out)
                out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)
            else:
                out = [self.model(data) for _ in range(iterations)]
                out = _stack_preds(out)
            return out

    def get_params(self):
        """
        Return the parameters to optimize.

        Returns:
            Config for parameters.
        """
        return self.model.parameters()

    def state_dict(self):
        """Get the state dict(s)."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load the model with `state_dict`."""
        self.model.load_state_dict(state_dict, strict=strict)

    def train(self):
        """Set the model in `train` mode."""
        self.model.train()

    def eval(self):
        """Set the model in `eval mode`."""
        self.model.eval()

    def reset_fcs(self):
        """Reset all torch.nn.Linear layers."""

        def reset(m):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        self.model.apply(reset)

    def reset_all(self):
        """Reset all *resetable* layers."""

        def reset(m):
            for m in self.model.modules():
                getattr(m, "reset_parameters", lambda: None)()

        self.model.apply(reset)

    def set_dataset_size(self, dataset_size: int):
        """
        Set state for dataset size. Useful for tracking.

        Args:
            dataset_size: Dataset state
        """
        self._active_dataset_size = dataset_size


def mc_inference(model, data, iterations, replicate_in_memory):
    if replicate_in_memory:
        input_shape = data.size()
        batch_size = input_shape[0]
        try:
            data = torch.stack([data] * iterations)
        except RuntimeError as e:
            raise RuntimeError(
                """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
            Use `replicate_in_memory=False` in order to reduce the memory requirements.
            Note that there will be some speed trade-offs"""
            ) from e
        data = data.view(batch_size * iterations, *input_shape[1:])
        try:
            out = model(data)
        except RuntimeError as e:
            raise RuntimeError(
                """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
            Use `replicate_in_memory=False` in order to reduce the memory requirements.
            Note that there will be some speed trade-offs"""
            ) from e
        out = map_on_tensor(lambda o: o.view([iterations, batch_size, *o.size()[1:]]), out)
        out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)
    else:
        out = [model(data) for _ in range(iterations)]
        if isinstance(out[0], Sequence):
            out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
        else:
            out = torch.stack(out, dim=-1)
    return out
