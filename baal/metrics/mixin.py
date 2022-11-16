from typing import Callable, Dict, Any, Optional, DefaultDict, Union

import numpy as np
from torchmetrics import Metric as TorchMetric

from baal.utils.metrics import Metrics


class MetricMixin:
    metrics: Dict[str, Union[Metrics, TorchMetric]]
    active_learning_metrics: DefaultDict[int, Dict[str, Any]]
    _active_dataset_size: int

    def active_step(self, dataset_size: Optional[int], metrics: Dict[str, Any]):
        """
        Log metrics at the end of the active learning step.

        Args:
            dataset_size: Current dataset size, if None, take state of `_active_dataset_size`.
            metrics: Metrics values

        Notes:
            Metrics can be overridden.
        """
        if dataset_size is None:
            dataset_size = self._active_dataset_size
        self.active_learning_metrics[dataset_size].update(metrics)

    def add_metric(self, name: str, initializer: Callable[[], Union[Metrics, TorchMetric]]):
        """
        Add a baal.utils.metric.Metrics or torchmetrics.Metric to the Model.

        Args:
            name (str): name of the metric.
            initializer (Callable): lambda to initialize a new instance of a
                                    Union[Metrics, TorchMetric].
        """
        self.metrics["test_" + name] = initializer()
        self.metrics["train_" + name] = initializer()

    def _reset_metrics(self, filter=""):
        """
        Reset all Metrics according to a filter.

        Args:
            filter (str): Only keep the metric if `filter` in the name.
        """
        for k, v in self.metrics.items():
            if filter in k:
                v.reset()

    def get_metrics(self, filter="") -> Dict[str, Any]:
        """
        Get all Metric values according to a filter.

        Args:
            filter (str): Only keep the metric if `filter` in the name.

        Returns:
            Dictionary with all values
        """

        def get_value(met: Union[Metrics, TorchMetric]):
            if isinstance(met, Metrics):
                return met.value
            val = met.compute().detach().cpu().numpy()
            if val.shape == ():
                val = val.item()
            return val

        metrics = {
            met_name: get_value(met) for met_name, met in self.metrics.items() if filter in met_name
        }
        if self._active_dataset_size != -1:
            # Add dataset size if it was ever set.
            metrics.update({"dataset_size": self._active_dataset_size})
        return metrics

    def _update_metrics(self, out, target, loss, filter=""):
        """
        Update all metrics.

        Args:
            out (Tensor): Prediction.
            target (Tensor): Ground truth.
            loss (Tensor): Loss from the criterion.
            filter (str): Only update metrics according to this filter.
        """
        for met_name, metric in self.metrics.items():
            if filter in met_name:
                if "loss" in met_name:
                    metric.update(loss)
                else:
                    metric.update(out, target)
