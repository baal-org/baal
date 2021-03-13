# BaaL cheat sheet

In the table below, we have a mapping between common equations and the BaaL API.

### Setup

Here are the types for all variables needed.

```python
model : torch.nn.Module
wrapper : baal.ModelWrapper
dataset: torch.utils.data_utils.Dataset 
bald = baal.active.heuristics.BALD()
entropy = baal.active.heuristics.Entropy()
```

We assume that `baal.bayesian.dropout.patch_module` has been applied to the model.

`model = baal.bayesian.dropout.patch_module(model)`

```eval_rst
.. csv-table:: BaaL cheat sheet
   :header: "Description", "Equation", "BaaL"
   :widths: 20, 20, 40

   "Bayesian Model Averaging", ":math:`\hat{T} = p(y \mid x, {\cal D})= \int p(y \mid x, \theta)p(\theta \mid D) d\theta)`", "`wrapper.predict_on_dataset(dataset, batch_size=B, iterations=I, use_cuda=True).mean(-1)`"
   "MC-Dropout", ":math:`T = \{p(y\mid x_j, \theta_i)\} \mid x_j \in {\cal D}' ,i \in \{1, \ldots, I\}`", "`wrapper.predict_on_dataset(dataset, batch_size=B, iterations=I, use_cuda=True)`"
   "BALD", ":math:`{\cal I}[y, \theta \mid x, {\cal D}] = {\cal H}[y \mid x, {\cal D}] - {\cal E}_{p(\theta \mid {\cal D})}[{\cal H}[y \mid x, \theta]]`", "`bald.get_uncertainties(T)`"
   "Entropy", ":math:`\sum_c \hat{T}_c \log(\hat{T}_c)`", "`entropy.get_uncertainties(T)`"

```

**Contributing**

If some equations are missing, please open a PR so that we can make this cheat sheet as useful as possible.

---