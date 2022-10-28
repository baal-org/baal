# Utilities


## Metrics

To work with `baal.modelwrapper.ModelWrapper`, we provide `Metrics`.

Starting with Baal 1.7.0, users can use [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/) as well.


### Examples

```python
from baal.modelwrapper import ModelWrapper
from baal.utils.metrics import Accuracy
from torchmetrics import F1Score

wrapper : ModelWrapper = ...
# You can add any metrics from `baal.utils.metrics`.
wrapper.add_metric(name='accuracy',initializer=lambda : Accuracy())
wrapper.add_metric(name='f1',initializer=lambda : F1Score())

# Metrics are automatically updated when training and evaluating.
wrapper.train_on_dataset(...)
wrapper.test_on_dataset(...)

print(wrapper.get_metrics())
"""
>>> {'dataset_size': 200,
    'test_accuracy': 0.2603,
    'test_f1': 0.1945,
    'test_loss': 2.1901,
    'train_accuracy': 0.3214,
    'train_f1': 0.2531,
    'train_loss': 2.1795}
"""

# Get metrics per dataset_size (state is kept for the entire loop.
print(wrapper.active_learning_metrics)
"""
>>> {200: {'dataset_size': 200,
    'test_accuracy': 0.26038339734077454,
    'test_loss': 2.190103769302368,
    'train_accuracy': 0.3214285671710968,
    'train_loss': 2.1795670986175537},
    ...
"""
```

### baal.utils.metrics

::: baal.utils.metrics