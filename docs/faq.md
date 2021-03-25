# Baal FAQ

If you have more questions, please submit an issue, and we will include it here!

## How to predict uncertainty per sample in a dataset

```python
model = YourModel()
# If not done already, you can wrap your model with our MCDropoutModule
model = MCDropoutModule(model)
dataset = YourDataset()
wrapper = ModelWrapper(model, criterion=None)

heuristic = BALD()

# This has a shape [iterations, len(dataset), num_classes, ...]
predictions = wrapper.predict_on_dataset(dataset, batch_size=32, iterations=20, use_cuda=True)
uncertainty = heuristic.get_uncertainties(predictions)
```

If your model or dataset is too large:

```python
pred_generator = wrapper.predict_on_dataset_generator(dataset, batch_size=32, iterations=20, use_cuda=True)
uncertainty = heuristic.get_uncertainties_generator(pred_generator)
```

## Does BaaL work on semantic segmentation?

Yes! See the example in `experiments/segmentation/unet_mcdropout_pascal.py`.

The key idea is to provide the Heuristic with a way to aggregate the uncertainties. In the case of semantic
segmentation, MC-Dropout will provide a distribution per pixel. To reduce this to a single uncertainty value,
you can provide `reduction` to the Heuristic with one of the following arguments:

* String (one of `'max'`, `'mean'`, `'sum'`)
* Callable, a function that will receive the uncertainty per pixel.

## Does BaaL work on NLP/TS/Tabular data?

BaaL is not task-specific, it can be used on a variety of domains and tasks. We are working toward more examples.

Bayesian active learning has been used for Text Classification and NER in [(Siddhant and Lipton, 2018)](http://zacklipton.com/media/papers/1808.05697.pdf).

## How to know if my model is calibrated

Baal uses the ECE to compute the calibration of a model. It is available throught: `baal.utils.metrics.ECE` and `baal.utils.metrics.ECE_PerCLs`, the latter providing the metrics per class.

You can add this metric to your model wrapper doing `ModelWrapper.add_metric('ece', lambda: ECE(n_bins=20))`

After training and testing, you can get your score with:
```
metrics = your_model.metrics
# Test ECE
metrics['test_ece'].value
# Train ECE
metrics['train_ece'].value
```

## What to do if my models/datasets don't fit in memory?

There is several ways to use Baal on large tasks.

* If MC sampling does not fit, you can use a for-loop instead.
    * Set ModelWrapper `replicate_in_memory=False`.
* If the size of the prediction does not fit.
    * Heuristics support generators
    * Use `ModelWrapper.predict_on_dataset_generator`


## How can I specify that a label is missing and how to label it.

The source of truth for what is labelled is the `ActiveLearningDataset.labelled` array.
This means that we will never train on a sample if it is not labelled according to this array.
This array determines the split between the labelled and unlabelled datasets.

```python
# Let ds = D, the entire dataset with labelled/unlabelled data.
ds = YourDataset()
al_dataset = ActiveLearningDataset(ds, ...)
# For convenience, let's label 10 samples at random.
# But you can provide the `labelled` array to ActiveLearningDataset
# if you already have labels.
al_dataset.label_randomly(10)
pool = al_dataset.pool
```

From a rigorous point of view: ``$`D = ds `$`` , ``$`D_L=al\_dataset `$`` and ``$`D_U = D \setminus D_L = pool `$``.
Then, we train our model on ``$`D_L `$`` and compute the uncertainty on ``$`D_U `$``. The most uncertains samples are labelled and added to ``$`D_L `$``, removed from ``$`D_U `$``.

Let a method `query_human` performs the annotations, we can label our dataset using indices relative to``$`D_U `$``. This assumes that your dataset class `YourDataset` has a method named `label` which has the following definition: `def label(self, idx, value)` where we give the label for index `idx`. There the index is not relative to the pool, so you don't have to worry about it.


#### Full example.

```python
# Some definitions
your_heuristic = BALD()
pool = active_dataset.pool
your_predictions = ModelWrapper.predict_on_dataset(pool, ...)
# The shape of `your_predictions` is [len(pool), n_classes, ..., iterations]
# Get the next batch of samples to label. Note: These indices are according to the pool.
ranks = your_heuristic(your_predictions)

# Now let's ask a human to label those samples.
labels = query_human(ranks, pool)

# To edit the dataset labels, you can now add those labels to your dataset. Still, the indices are according to the pool.
active_dataset.label(ranks, labels)
```


## Tips & Trick for a successful active learning experiment

Many of these tips can be found in our paper [Bayesian active learning for production](https://arxiv.org/abs/2006.09916).

#### Remove data augmentation when computing uncertainty

You can specify which variables to override when creating the unlabelled pool using the `pool_specifics` argument.
```python
from torchvision import transforms
transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ]) 
test_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor()
                ]) 
                
your_dataset = ADataset(transform=transform)
active_dataset = ActiveLearningDataset(your_dataset, pool_specifics={'transform':test_transform})

# active_dataset will use data augmentation
# the pool will use the `test_transform`
pool = active_dataset.pool
```

#### Reset the model to its original weights (Gal et al. 2017)

```python
# Make a deep copy of the initial weights
initial_weights = copy.deepcopy(model.state_dict())
loop = ActiveLearningLoop(...)

for al_step in range(NUM_AL_STEP):
    # Reset the weights to its initial value
    model.load_state_dict(initial_weights)
    # Train to convergence
    model.train_on_dataset(...)
    # Test on the validation set.
    model.test_on_dataset(...)
    # Label the next set of labels.
    loop.step()
    
```

#### Use Bayesian model average when testing.

When using MC-Dropout, or any other Bayesian methods, you will want to compute the Bayesian model average (BMA) at test time too.

To do so, you can specify the `average_predictions` parameters in `ModelWrapper.test_on_dataset`. The prediction will be averaged over `iterations` stochastic predictions. 

This will slightly increase the ECE of your model and will improve the predictive performance as well.

#### Compute uncertainty on a subset of the unlabelled pool

Predicting on the unlabelled pool is the most time consuming part of active learning, especially in expensive tasks such as segmentation.

Our work shows that predicting on a random subset of the pool is as effective as the full prediction. BaaL supports this features throught the `max_samples` argument in `ActiveLearningPool`.
