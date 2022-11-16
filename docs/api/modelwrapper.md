# ModelWrapper

`ModelWrapper` is an object similar to `keras.Model` that trains, test and predict on datasets.

Using our wrapper makes it easier to do Monte-Carlo sampling with the `iterations` parameters.
Another optimization that we do is that instead of using a for-loop to perform MC sampling, we stack examples.

### Example

```python
from baal.modelwrapper import ModelWrapper
from baal.active.dataset import ActiveLearningDataset
from torch.utils.data import Dataset

# You define ModelWrapper with a Pytorch model and a criterion.
wrapper = ModelWrapper(model=your_model, criterion=your_criterion)

# Assuming you have your ActiveLearningDataset ready,
al_dataset: ActiveLearningDataset = ...
test_dataset: Dataset = ...

train_history = wrapper.train_on_dataset(al_dataset, optimizer=your_optimizer, batch_size=32, epoch=10, use_cuda=True)
# We can also use BMA during test time using `average_predictions`.
test_values = wrapper.test_on_dataset(test_dataset, average_predictions=20, **kwargs)

# We use Monte-Carlo sampling using the `iterations` arguments.
predictions = wrapper.predict_on_dataset(al_dataset.pool, iterations=20, **kwargs)
predictions.shape
# > [len(al_dataset.pool), num_outputs, 20]

```

### API

### baal.ModelWrapper

::: baal.ModelWrapper