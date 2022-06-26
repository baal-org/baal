from collections import defaultdict
from numbers import Number
from typing import Callable, Dict, Any, Optional, DefaultDict

from baal.utils.metrics import Metrics


class MetricMixin:
    metrics: Dict[str, Metrics]
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

    def add_metric(self, name: str, initializer: Callable):
        """
        Add a baal.utils.metric.Metric to the Model.

        Args:
            name (str): name of the metric.
            initializer (Callable): lambda to initialize a new instance of a
                                    baal.utils.metrics.Metric object.
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
        return {met_name: met.value for met_name, met in self.metrics.items() if filter in met_name}

    def _update_metrics(self, out, target, loss, filter=""):
        """
        Update all metrics.

        Args:
            out (Tensor): Prediction.
            target (Tensor): Ground truth.
            loss (Tensor): Loss from the criterion.
            filter (str): Only update metrics according to this filter.
        """
        for k, v in self.metrics.items():
            if filter in k:
                if "loss" in k:
                    v.update(loss)
                else:
                    v.update(out, target)
