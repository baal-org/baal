import torch
from typing import Optional, List, Sequence
import numpy as np
from transformers import Trainer

from baal.utils.array_utils import stack_in_memory
from baal.utils.iterutils import map_on_tensor


class BaalHuggingFaceTrainer(Trainer):


    def predict_on_dataset_generator(self,
                                     dataset,
                                     iterations: int = 1,
                                     half: bool = False,
                                     ignore_keys: Optional[List[str]] = None):

        dataloader = self.get_eval_dataloader(dataset)

        model = self.model

        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        model.eval()
        # with torch.no_grad():
        #     if cuda:
        #         data = to_cuda(data)
        for step, inputs in enumerate(dataloader):
            inputs = map_on_tensor(lambda d: stack_in_memory(d, iterations), inputs)
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
            ignore_keys Optional[List[str]]: A list of keys in the output of your model
                (if it is a dictionary) that should be ignored when gathering predictions.
        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Array [n_samples, n_outputs, ..., n_iterations].
        """
        preds = list(self.baal_predict_generator(dataset=dataset,
                                                 iterations=iterations,
                                                 half=half,
                                                 ignore_keys=ignore_keys))

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]
