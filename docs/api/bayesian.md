# Bayesian deep learning

In Bayesian active learning, we draw from the posterior distribution to estimate uncertainty.

## Example

```python
from baal.bayesian.dropout import MCDropoutModule, patch_module
from baal.bayesian.weight_drop import MCDropoutConnectModule

model: nn.Module
# To make Dropout layers always on
model = MCDropoutModule(model)
# or
model = patch_module(model)


# To use MC-Dropconnect on all linear layers
model = MCDropoutConnectModule(model, layers=["Linear"], weight_dropout=0.5)
```


## API

```eval_rst
.. autoclass:: baal.bayesian.dropout.MCDropoutModule
    :members: __init__

..autoclass:: baal.bayesian.weight_drop.MCDropoutConnectModule
    :members: __init__

```