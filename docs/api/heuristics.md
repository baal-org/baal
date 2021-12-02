# Heuristics

Heuristics take a set of predictions and output an uncertainty value.
They are agnostic to the method used for predicting, so they work with MC sampling and Ensembles.

### Example

Using BALD, we can compute the uncertainty of many predictions.

```python
import numpy as np
from baal.active.heuristics import BALD

# output from ModelWrapper.predict_on_dataset with shape [1000, num_classes, 20]
predictions: np.ndarray = ... 

# To get the full uncertainty score
uncertainty = BALD().compute_score(predictions)

# To get ranks
most_uncertain = BALD()(predictions)

# If you wish to mix BALD and Uniform sampling,
# you can modify the `shuffle_prop` parameter.
BALD(shuffle_prop=0.1)

# When working with Sequence or Segmentation models, you can specify how to aggregate
# values using the "reduction" parameter.
BALD(reduction="mean")

```


### API

```eval_rst
.. autoclass:: baal.active.heuristics.AbstractHeuristic
    :members:

.. autoclass:: baal.active.heuristics.BALD

.. autoclass:: baal.active.heuristics.Random

.. autoclass:: baal.active.heuristics.Entropy
```