# Active Learning Experiment

In this module, we find all the utilities to do active learning.
Baal takes care of the dataset split between labelled and unlabelled examples.
It also takes care of the active learning loop:

1. Train the model on the label set.
2. Evaluate the model on the evaluation set.
3. Predict on the unlabelled examples.
4. Label the most uncertain examples.
5. Stop the experiment if finished.

### Example

```python
from baal.active.dataset import ActiveLearningDataset
from baal.experiments.base import ActiveLearningExperiment
al_dataset = ActiveLearningDataset(your_dataset)

# To start, we can select 1000 random examples to be labelled
al_dataset.label_randomly(1000)

experiment = ActiveLearningExperiment(
    trainer=..., # Huggingface or ModelWrapper to train
    al_dataset=al_dataset, # Active learning dataset
    eval_dataset=..., # Evaluation Dataset
    heuristic=BALD(), # Uncertainty heuristic to use
    query_size=100, # How many items to label per round.
    iterations=20, # How many MC sampling to perform per item.
    pool_size=None, # Optionally limit the size of the unlabelled pool.
    criterion=None # Stopping criterion for the experiment.
)
experiment.start()
```

### API

### baal.experiments.base.ActiveLearningExperiment
::: baal.experiments.base.ActiveLearningExperiment
