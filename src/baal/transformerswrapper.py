import torch
from typing import Optional, List, Sequence
import numpy as np

# These packages are optional and not needed for BaaL main package.
# You can have access to `datasets` and `transformers` if you install
# BaaL with --dev setup.
from transformers import Trainer

from baal.utils.array_utils import stack_in_memory
from baal.utils.iterutils import map_on_tensor, map_on_dict_elements


class BaalHuggingFaceTrainer(Trainer):
    """
    The purpose of this wrapper is to provide extra capabilities for HuggingFace Trainer(
    https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html), so that it can
    output several forward pass for samples in prediction time and hence be able to work with baal.
    """

    def predict_on_dataset_generator(self,
                                     dataset,
                                     iterations: int = 1,
                                     half: bool = False,
                                     ignore_keys: Optional[List[str]] = None):
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

        model = self.model

        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        for step, inputs in enumerate(dataloader):
            inputs = map_on_dict_elements(lambda element: map_on_tensor(
                lambda d: stack_in_memory(d, iterations), element), inputs)
            _, out, _ = self.prediction_step(model,
                                             inputs,
                                             prediction_loss_only=False,
                                             ignore_keys=ignore_keys)

            out = map_on_tensor(lambda o: o.view([iterations, -1, *o.size()[1:]]), out)
            out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)
            out = map_on_tensor(lambda x: x.detach(), out)
            if half:
                out = map_on_tensor(lambda x: x.half(), out)
            yield map_on_tensor(lambda x: x.cpu().numpy(), out)

    def predict_on_dataset(self,
                           dataset,
                           iterations: int = 1,
                           half: bool = False,
                           ignore_keys: Optional[List[str]] = None):

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
        preds = list(self.predict_on_dataset_generator(dataset=dataset,
                                                       iterations=iterations,
                                                       half=half,
                                                       ignore_keys=ignore_keys))

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def load_state_dict(self, state_dict, strict=True):
        """Load the model with `state_dict`."""
        self.model.load_state_dict(state_dict, strict=strict)
