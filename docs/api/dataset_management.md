# Active Learning Dataset

In this module, we find all the utilities to manage data when performing active learning.

Baal takes care of the dataset split between labelled and unlabelled examples.
### Example

```python
from baal.active.dataset import ActiveLearningDataset
al_dataset = ActiveLearningDataset(your_dataset)

# To start, we can select 1000 random examples to be labelled
al_dataset.label_randomly(1000)

# Our training set is now 1000
len(al_dataset)

# We can label examples by their indices.
al_dataset.label([32, 10, 4])

# Our dataset length is now 1003.
len(al_dataset)

# At initialization, we can also swap attributes for the pool.
al_dataset = ActiveLearningDataset(your_dataset, pool_specifics={"transform": None})
assert al_dataset.pool.transform is None
```

### API

### baal.active.ActiveLearningDataset
::: baal.active.ActiveLearningDataset

### baal.active.FileDataset
::: baal.active.FileDataset