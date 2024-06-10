from typing import Optional, List, Sequence, Union

import numpy as np
import structlog
import torch
from numpy._typing import NDArray
from torch import nn
from transformers import PreTrainedModel, TrainingArguments
from transformers.utils.logging import tqdm

from baal.utils.warnings import raise_warnings_cache_replicated

# These packages are optional and not needed for BaaL main package.
try:
    from transformers.trainer import Trainer
except ImportError:
    raise ImportError(
        "`transformers` library is required to use this module."
        " Please do `pip install baal[nlp]`"
    )

from baal.utils.array_utils import stack_in_memory
from baal.utils.iterutils import map_on_tensor

log = structlog.get_logger("baal")


def _stack_preds(out):
    if isinstance(out[0], Sequence):
        out = [torch.stack(ts, dim=0) for ts in zip(*out)]
    else:
        out = torch.stack(out, dim=0)
    return out


class BaalTransformersTrainer(Trainer):
    """
    The purpose of this wrapper is to provide extra capabilities for HuggingFace Trainer, so that
    it can output several forward pass for samples in prediction time and hence be able to work with
    baal. For a more detailed description of the arguments refer to (
    https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html)

    Args:
        model (transformers.PreTrainedModel): The model to train, evaluate or use for predictions.
        replicate_in_memory: If True, will perform MC-Dropout in a single forward pass.
            It is faster, but more memory expensive. Default: True.
        data_collator (Optional(Callable)): The function to use to from a batch.
        train_dataset (Optional(torch.utils.data.Dataset)): The dataset to use for training.
        eval_dataset (Optional(torch.utils.data.Dataset)): The dataset to use for evaluation.
        tokenizer (Optional(transformers.PreTrainedTokenizer)): a tokenizer provided by huggingface.
        model_init (Optional(Dict)): Model initial weights for fine tuning.
        compute_metrics (Optional(Callable[[EvalPrediction], Dict])): The function that will be
            used to compute metrics at evaluation.
        callbacks (Optional(List[transformers.TrainerCallback])): A list of callbacks to customize
            the training loop.
        optimizers (Optional(Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR])):
            A tuple containing the optimizer and the scheduler to use.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        replicate_in_memory=True,
        **kwargs
    ):
        self.replicate_in_memory = replicate_in_memory
        super().__init__(model=model, args=args, **kwargs)
        raise_warnings_cache_replicated(model, replicate_in_memory)

    def predict_on_dataset_generator(
        self,
        dataset,
        iterations: int = 1,
        half: bool = False,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        Use the model to predict on a dataset `iterations` time.

        Args:
            dataset (Dataset): Dataset to predict on.
            iterations (int): Number of iterations per sample.
            half (bool): If True use half precision.
            ignore_keys (Optional[List[str]]): A list of keys in the output of your model
                (if it is a dictionary) that should be ignored when gathering predictions.
        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Generators [batch_size, n_classes, ..., n_iterations].
        """

        dataloader = self.get_eval_dataloader(dataset)
        log.info("Start Predict", dataset=len(dataset))

        model = self.model

        model.eval()
        for step, inputs in enumerate(tqdm(dataloader)):
            if self.replicate_in_memory:
                try:
                    # We perform MC-Dropout in a single pass, fast, but memory intensive.
                    inputs = map_on_tensor(
                        lambda element: map_on_tensor(
                            lambda d: stack_in_memory(d, iterations), element
                        ),
                        inputs,
                    )
                    _, out, _ = self.prediction_step(
                        model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        raise RuntimeError(
                            """CUDA ran out of memory while Baal tried to replicate data. See the exception above.
                        Use `replicate_in_memory=False` in order to reduce the memory requirements.
                        Note that there will be some speed trade-offs"""
                        ) from e
                    raise e
                out = map_on_tensor(lambda o: o.view([iterations, -1, *o.size()[1:]]), out)
            else:
                # We perform a forward pass `iterations` time. Slower, but memory efficient.
                out = [
                    self.prediction_step(
                        model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
                    )[1]
                    for _ in range(iterations)
                ]
                out = _stack_preds(out)
            # Swap axes to match Baal [Batch_size, Classes, ..., Iterations]
            out = map_on_tensor(lambda o: o.permute(1, *range(3, o.ndimension()), 2, 0), out)
            out = map_on_tensor(lambda x: x.detach(), out)
            if half:
                out = map_on_tensor(lambda x: x.half(), out)
            yield map_on_tensor(lambda x: x.cpu().numpy(), out)

    def predict_on_dataset(
        self,
        dataset,
        iterations: int = 1,
        half: bool = False,
        ignore_keys: Optional[List[str]] = None,
    ) -> Union[NDArray, List[NDArray]]:

        """
        Use the model to predict on a dataset `iterations` time.

        Args:
            dataset (Dataset): Dataset to predict on.
            iterations (int): Number of iterations per sample.
            half (bool): If True use half precision.
            ignore_keys (Optional[List[str]]): A list of keys in the output of your model
                (if it is a dictionary) that should be ignored when gathering predictions.
        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Array [n_samples, n_outputs, ..., n_iterations].
        """
        preds = list(
            self.predict_on_dataset_generator(
                dataset=dataset, iterations=iterations, half=half, ignore_keys=ignore_keys
            )
        )

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def load_state_dict(self, state_dict, strict=True):
        """Load the model with `state_dict`."""
        self.model.load_state_dict(state_dict, strict=strict)
