# From https://github.com/pytorch/ignite with slight changes
import dataclasses
import math
import warnings
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, auc

from baal.utils.array_utils import to_prob


def transpose_and_flatten(input):
    if input.dim() > 2:
        input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
    return input


class Metrics:
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
        warnings.warn("`avg` is deprecated, please use `value`.", DeprecationWarning)
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
        self.tp, self.samples, self.conf_agg = None, None, None
        super().__init__(average=False)

    def update(self, output=None, target=None):
        """
        Updating the true positive (tp) and number of samples in each bin.

        Args:
            output (tensor): logits or predictions of model
            target (tensor): labels
        """

        output = transpose_and_flatten(output).detach().cpu().numpy()
        target = target.view([-1]).detach().cpu().numpy()
        output = to_prob(output)

        # this is to make sure handling 1.0 value confidence to be assigned to a bin
        output = np.clip(output, 0, 0.9999)

        for pred, t in zip(output, target):
            conf, p_cls = pred.max(), pred.argmax()

            bin_id = int(math.floor(conf * self.n_bins))
            self.samples[bin_id] += 1
            self.tp[bin_id] += int(p_cls == t)
            self.conf_agg[bin_id] += conf

    def _acc(self):
        return self.tp / np.maximum(1, self.samples)

    def calculate_result(self):
        n = self.samples.sum()
        average_confidence = self.conf_agg / np.maximum(self.samples, 1)
        return ((self.samples / n) * np.abs(self._acc() - average_confidence)).sum()

    @property
    def value(self):
        return self.calculate_result()

    def plot(self, pth=None):
        """
        Plot each bins, ideally this would be a diagonal line.

        Args:
            pth (str): if provided the figure will be saved under the given path
        """
        import matplotlib.pyplot as plt

        # Plot the ECE
        plt.bar(np.linspace(0, 1, self.n_bins), self._acc(), align="edge", width=0.1)
        plt.plot([0, 1], [0, 1], "--", color="tab:gray")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel("Accuracy")
        plt.xlabel("Uncertainty")
        plt.grid()

        if pth:
            plt.savefig(pth)
            plt.close()
        else:
            plt.show()

    def reset(self):
        self.tp = np.zeros([self.n_bins])
        self.samples = np.zeros([self.n_bins])
        self.conf_agg = np.zeros([self.n_bins])


class ECE_PerCLs(Metrics):
    """
    Expected Calibration Error (ECE)

    Args:
        n_cls (int): number of existing target classes
        n_bins (int): number of bins to discretize the uncertainty.

    References:
        https://arxiv.org/pdf/1706.04599.pdf
    """

    def __init__(self, n_cls, n_bins=10, **kwargs):
        self.n_bins = n_bins
        self.n_cls = n_cls
        self.samples = np.zeros([self.n_cls, self.n_bins], dtype=int)
        self.tp = np.zeros([self.n_cls, self.n_bins], dtype=int)
        self.conf_agg = np.zeros([self.n_cls, self.n_bins])
        super().__init__(average=False)

    def update(self, output=None, target=None):
        """
        Updating the true positive (tp) and number of samples in each bin.

        Args:
            output (tensor): logits or predictions of model
            target (tensor): labels
        """
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        output = to_prob(output)

        # this is to make sure handling 1.0 value confidence to be assigned to a bin
        output = np.clip(output, 0, 0.9999)

        for pred, t in zip(output, target):
            t = int(t)  # Force the conversion
            conf, p_cls = pred.max(), pred.argmax()
            bin_id = int(math.floor(conf * self.n_bins))
            self.samples[t, bin_id] += 1
            self.tp[t, bin_id] += int(p_cls == t)
            self.conf_agg[t, bin_id] += conf

    def _acc(self):
        accuracy_per_class = np.zeros([self.n_cls, self.n_bins], dtype=float)
        for cls in range(self.n_cls):
            accuracy_per_class[cls, :] = self.tp[cls, :] / np.maximum(1, self.samples[cls, :])
        return accuracy_per_class

    def calculate_result(self):
        """calculates the ece per class.

        Returns:
            ece (nd.array): ece value per class
        """
        accuracy = self._acc()
        ece = np.zeros([self.n_cls])
        for cls in range(self.n_cls):
            n = self.samples[cls, :].sum()
            if n == 0:
                ece[cls] = 0
            else:
                bin_confidence = self.conf_agg[cls] / np.maximum(1, self.samples[cls])
                diff_accuracy = np.abs(accuracy[cls, :] - bin_confidence)
                ece[cls] = ((self.samples[cls, :] / n) * diff_accuracy).sum()
        return ece

    @property
    def value(self):
        return self.calculate_result()

    def plot(self, pth=None):
        """
        Plot each bins, ideally this would be a diagonal line.

        Args:
            pth (str): if provided the figure will be saved under the given path
        """
        import matplotlib.pyplot as plt

        accuracy = self._acc()
        # Plot the ECE
        fig, axs = plt.subplots(self.n_cls)
        for cls in range(self.n_cls):
            axs[cls].bar(np.linspace(0, 1, self.n_bins), accuracy[cls, :], align="edge", width=0.1)
            axs[cls].plot([0, 1], [0, 1], "--", color="tab:gray")
            axs[cls].set_ylim(0, 1)
            axs[cls].set_xlim(0, 1)
            axs[cls].set_ylabel("Accuracy")
            axs[cls].set_xlabel("Uncertainty")
            axs[cls].grid()
        if pth:
            plt.savefig(pth)
            plt.close()
        else:
            plt.show()

    def reset(self):
        self.tp = np.zeros([self.n_cls, self.n_bins])
        self.samples = np.zeros([self.n_cls, self.n_bins])
        self.conf_agg = np.zeros([self.n_cls, self.n_bins])


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
    """computes the top first and top five accuracy for the model batch by
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
        acc: torch.Tensor = self.accuracy
        return acc


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
            target.reshape([-1]).astype(int),
            output.reshape([-1]).astype(int),
            labels=np.arange(self.class_data.shape[0]),
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
        return {"accuracy": acc, "precision": precision, "recall": recall}


@dataclasses.dataclass
class Report:
    tp: int = 0
    fp: int = 0
    fn: int = 0


class PRAuC(Metrics):
    """
    Precision-Recall Area under the curve.

    Args:
        num_classes (int): Number of classes
        n_bins (int): number of confidence threshold to evaluate on.
        average (bool): If true will return the mean AuC of all classes.
    """

    def __init__(self, num_classes, n_bins, average):
        self.num_classes = num_classes
        self.threshold = np.linspace(0.02, 0.99, n_bins)
        self._data = defaultdict(lambda: defaultdict(lambda: Report()))
        super().__init__(average)

    def reset(self):
        self._data.clear()

    def update(self, output=None, target=None):
        """
        Update the confusion matrice according to output and target.

        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        output = to_prob(output)

        assert output.ndim > target.ndim, "Only multiclass classification is supported."
        for cls in range(self.num_classes):
            target_cls = (target == cls).astype(np.int8)
            for th in self.threshold:
                report = self._make_report(output[:, cls, ...], target_cls, th)
                self._data[cls][th].fp += report.fp
                self._data[cls][th].tp += report.tp
                self._data[cls][th].fn += report.fn

    def _make_report(self, output, target, threshold):
        output = (output > threshold).astype(np.int8)
        output = output.reshape([-1])
        target = target.reshape([-1])
        _, fp, fn, tp = confusion_matrix(target, output, labels=[0, 1]).ravel()
        return Report(tp=tp, fp=fp, fn=fn)

    @property
    def value(self):
        result = []
        for cls in range(self.num_classes):
            precisions = np.array([r.tp / max(1, r.tp + r.fp) for r in self._data[cls].values()])
            recalls = np.array([r.tp / max(1, r.tp + r.fn) for r in self._data[cls].values()])
            idx = np.argsort(recalls)
            precisions = precisions[idx]
            recalls = recalls[idx]

            result.append(auc(recalls, precisions))
        if self._average:
            return np.mean(result)
        return result
