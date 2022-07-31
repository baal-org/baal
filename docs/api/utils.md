# Utilities


## Metrics

To work with `baal.modelwrapper.ModelWrapper`, we provide `Metrics`.


### Examples

```python
from baal.modelwrapper import ModelWrapper
from baal.utils.metrics import Accuracy

wrapper : ModelWrapper = ...
# You can add any metrics from `baal.utils.metrics`.
wrapper.add_metric(name='accuracy',initializer=lambda : Accuracy())

# Metrics are automatically set when training and evaluating.
wrapper.train_on_dataset(...)
wrapper.test_on_dataset(...)

print(wrapper.get_metrics())
"""
>>> {'dataset_size': 200,
    'test_accuracy': 0.26038339734077454,
    'test_loss': 2.190103769302368,
    'train_accuracy': 0.3214285671710968,
    'train_loss': 2.1795670986175537}
...
"""
```

```eval_rst
.. automodule:: baal.utils.metrics
    :members:
```