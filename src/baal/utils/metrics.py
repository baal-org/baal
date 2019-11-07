# From https://github.com/pytorch/ignite with slight changes
import math
import warnings

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class Metrics(object):
    """
    metric is an abstract class.
    Args:
        average (bool): a way to output one single value for metrics
                        that are calculated in several trials.
    """

    def __init__(self, average=True, **kwargs):
        self._average = average
        self.eps = 1e-20
        self.reset()
        self.result = torch.FloatTensor()

    def reset(self):
        """Reset the private values of the class."""
        raise NotImplementedError

    def update(self, output=None, target=None):
        """
        Main calculation of the metric which updated the private values respectively.

        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        raise NotImplementedError

    def calculate_result(self):
        """calculate the final values when the epoch/batch loop
        is finished.
        """
        raise NotImplementedError

    @property
    def avg(self):
        warnings.warn('`avg` is deprecated, please use `value`.', DeprecationWarning)
        return self.value

    @property
    def value(self):
        """output the metric results (array shape) or averaging
        out over the results to output one single float number.

        Returns:
            result (np.array / float): final metric result

        """
        self.result = torch.FloatTensor(self.calculate_result())
        if self._average and self.result.numel() == self.result.size(0):
            return self.result.mean(0).cpu().numpy().item()
        elif self._average:
            return self.result.mean(0).cpu().numpy()
        else:
            return self.result.cpu().numpy()

    @property
    def standard_dev(self):
        """Return the standard deviation of the metric."""
        result = torch.FloatTensor(self.calculate_result())
        if result.numel() == result.size(0):
            return result.std(0).cpu().numpy().item()
        else:
            return result.std(0).cpu().numpy()

    def __str__(self):
        val = self.value
        std = self.standard_dev
        if isinstance(val, np.ndarray):
            return ", ".join(f"{v:.3f}±{s:.3f}" for v, s in zip(val, std))
        else:
            return f"{val:.3f}±{std:.3f}"


class ECE(Metrics):
    """
    Expected Calibration Error (ECE)

    Args:
        n_bins (int): number of bins to discretize the uncertainty.

    References:
        https://arxiv.org/pdf/1706.04599.pdf
    """

    def __init__(self, n_bins=10, **kwargs):
        self.n_bins = n_bins
        self.tp, self.samples = None, None
        super().__init__(average=False)

    def update(self, output=None, target=None):
        for pred, t in zip(output, target):
            conf, p_cls = pred.max(), pred.argmax()
            bin_id = int(math.floor(conf / (1.0 / self.n_bins)))
            self.samples[bin_id] += 1
            self.tp[bin_id] += int(p_cls == t)

    def _acc(self):
        return self.tp / np.maximum(1, self.samples)

    def calculate_result(self):
        n = self.samples.sum()
        bin_confs = np.linspace(0, 1, self.n_bins)
        return ((self.samples / n) * np.abs(self._acc() - bin_confs)).sum()

    @property
    def value(self):
        return self.calculate_result()

    def plot(self):
        """ Plot each bins, ideally this would be a diagonal line."""
        import matplotlib.pyplot as plt

        # Plot the ECE
        plt.bar(np.linspace(0, 1, self.n_bins), self._acc(), align='edge', width=0.1)
        plt.plot([0, 1], [0, 1], '--', color='tab:gray')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel('Accuracy')
        plt.xlabel('Uncertainty')
        plt.grid()
        plt.show()

    def reset(self):
        self.tp = np.zeros([self.n_bins])
        self.samples = np.zeros([self.n_bins])


class Loss(Metrics):
    """
    Args:
        average (bool): a way to output one single value for metrics
                        that are calculated in several trials.
    """

    def __init__(self, average=True, **kwargs):
        super().__init__(average=average)

    def reset(self):
        self.loss = list()

    def update(self, output=None, target=None):
        self.loss.append(output)

    def calculate_result(self):
        return self.loss


class Accuracy(Metrics):
    """ computes the top first and top five accuracy for the model batch by
    batch.
    Args:
        average (bool): a way to output one single value for metrics
                        that are calculated in several trials.
        topk (tuple): the value of k for calculating the topk accuracy.
    """

    def __init__(self, average=True, topk=(1,), **kwargs):
        super().__init__(average=average)

        self.topk = topk
        self.maxk = max(topk)

    def reset(self):
        self.accuracy = torch.FloatTensor()

    def update(self, output=None, target=None):
        """
        Update TP and support.

        Args:
            output (tensor): predictions of model
            target (tensor): labels

        Raises:
            ValueError if the first dimension of output and target don't match.
        """
        batch_size = target.shape[0]
        if not output.shape[0] == target.shape[0]:
            raise ValueError(
                f"Sizes of the output ({output.shape[0]}) and target "
                "({target.shape[0]}) don't match."
            )
        dim = 1
        predicted_indices = output.topk(self.maxk, dim, largest=True, sorted=True)[1]

        correct = predicted_indices.eq(target.view(-1, 1).expand_as(predicted_indices))

        topk_acc = []
        for k in self.topk:
            correct_k = correct[:, :k].contiguous().view(-1).float().sum()
            topk_acc.append(float(correct_k.mul_(1.0 / batch_size)))

        if len(self.accuracy) == 0:
            self.accuracy = torch.FloatTensor(topk_acc).unsqueeze(0)
        else:
            self.accuracy = torch.cat(
                [self.accuracy, torch.FloatTensor(topk_acc).unsqueeze(0)], dim=0
            )

    def calculate_result(self) -> torch.Tensor:
        return self.accuracy


class Precision(Metrics):
    """computes the precision for each class over epochs.
    Args:
        num_classes (int): number of classes.
        average (bool): a way to output one single value for metrics
                        that are calculated in several trials.
    """

    def __init__(self, num_classes: int, average=True, **kwargs):
        self.n_class = num_classes
        super().__init__(average=average)
        self._true_positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._positives = torch.zeros([self.n_class], dtype=torch.float32)

    def reset(self):
        self._true_positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._positives = torch.zeros([self.n_class], dtype=torch.float32)

    def update(self, output=None, target=None):
        """
        Update tp, fp and support acoording to output and target.

        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        # (batch, 1)
        target = target.view(-1)

        # (batch, nclass)
        indices = torch.argmax(output, dim=1).view(-1)

        output = indices.type_as(target)
        correct = output.eq(target.expand_as(output))

        # Convert from int cuda/cpu to double cpu
        for class_index in target:
            self._positives[class_index] += 1
        for class_index in indices[(correct == 1).nonzero()]:
            self._true_positives[class_index] += 1

    def calculate_result(self):
        result = self._true_positives / self._positives

        # where the class never was shown in targets
        result[result != result] = 0

        return result


class ClassificationReport(Metrics):
    """
    Compute a classification report as a metric.

    Args:
        num_classes (int): the number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(average=False)

    def reset(self):
        self.class_data = np.zeros([self.num_classes, self.num_classes])

    def update(self, output=None, target=None):
        """
        Update the confusion matrice according to output and target.

        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        if output.ndim > target.ndim:
            # Argmax not done
            output = output.argmax(1)  # 1 is always our class axis.

        self.class_data += confusion_matrix(
            target.reshape([-1]).astype(np.int), output.reshape([-1]).astype(np.int),
            labels=np.arange(self.class_data.shape[0])
        )

    @property
    def value(self):
        print("\n" + str(self.class_data))
        fp = self.class_data.sum(axis=0) - np.diag(self.class_data)
        fn = self.class_data.sum(axis=1) - np.diag(self.class_data)
        tp = np.diag(self.class_data)
        tn = self.class_data.sum() - (fp + fn + tp)
        acc = (tp + tn) / np.maximum(1, tp + fp + fn + tn)
        precision = tp / np.maximum(1, tp + fp)
        recall = tp / np.maximum(1, tp + fn)
        return {'accuracy': acc, 'precision': precision, 'recall': recall}
