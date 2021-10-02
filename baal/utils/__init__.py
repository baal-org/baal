from typing import Optional

from baal.utils import metrics


def get_metric(
    name: str, num_classes: Optional[int] = None, average: bool = True, **kwargs
) -> metrics.Metrics:
    """
    Create an heuristic object from the name.

    Args:
        name (str): Name of the metric.
        num_classes (Optional[int]): Number of outputs
        average (bool): give an averaged value for the whole epoch or a list of values.

    Returns:
        Metrics object.
    """
    metric: metrics.Metrics = {
        "loss": metrics.Loss,
        "accuracy": metrics.Accuracy,
        "precision": metrics.Precision,
        "classification_report": metrics.ClassificationReport,
    }[name](num_classes=num_classes, average=average, **kwargs)
    return metric
