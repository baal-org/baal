import torch

from baal.modelwrapper import ModelWrapper


class DropConWrapper(ModelWrapper):
    """
    class to make make `prediction_on_batch`
    do loops instead of stacking data.
    Args:
        model (torch.nn.Module): pytorch model
        criterion (torch.nn.Module): pytorch criterion
    """
    def __init__(self, model, criterion):
        super(DropConWrapper, self).__init__(model=model, criterion=criterion)

    def predict_on_batch(self, data, iterations=1, cuda=False):
        """
        Get the model's prediction on a batch.

        Args:
            data (Tensor): the model input
            iterations (int): number of prediction to perform.
            cuda (bool): use cuda or not

        Returns:
            Tensor, the loss computed from the criterion.
                    shape = {batch_size, nclass, n_iteration}
        """
        with torch.no_grad():
            if cuda:
                data = data.cuda()
            out = []
            for _ in range(iterations):
                out.append(self.model(data))
            out = torch.stack(out, dim=-1)
            return out
