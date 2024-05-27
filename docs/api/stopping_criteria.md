# Stopping Criteria

Stopping criterion are used to determine when to stop your active learning experiment.

Their usage are simple, but best put in practice with `ActiveExperiment`.

**Example**
```python
from baal.active.stopping_criteria import LabellingBudgetStoppingCriterion
from baal.active.dataset import ActiveLearningDataset

al_dataset: ActiveLearningDataset = ... # len(al_dataset) == 10
criterion = LabellingBudgetStoppingCriterion(al_dataset, labelling_budget=100)

assert not criterion.should_stop({}, [])

# len(al_dataset) == 60
al_dataset.label_randomly(50)
assert not criterion.should_stop({}, [])

# len(al_dataset) == 110, budget exhausted! We've labelled 100 items.
al_dataset.label_randomly(50)
assert criterion.should_stop({}, [])
```


### API

### baal.active.stopping_criteria

::: baal.active.stopping_criteria.LabellingBudgetStoppingCriterion

::: baal.active.stopping_criteria.LowAverageUncertaintyStoppingCriterion

::: baal.active.stopping_criteria.EarlyStoppingCriterion