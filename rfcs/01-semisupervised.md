# Semi-supervised learning support

---
| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | 1|
| **Author(s)** | @Dref360 |
| **Sponsor**   | None                 |
| **Updated**   | 2020-02-27                                           |


---

## Summary
The goal of this feature is to remove the burden required to perform semi-supervised learning.
Semi-supervised learning is widely popular in active learning and BaaL doesn't support it.
At first, we will **not** support label propagation, but it can be done in the future.

## Motivation

Many papers propose to combine active learning and semi-supervised learning. Currently, it is not easy to perform SS learning with BaaL. One needs to know the internal representation of ActiveLearningDataset to do so.
This RFC proposes a new API to make this easier.


## Guide-level explanation
 
I wish to do something similar to:
```python
al_dataset = ActiveLearningDataset(...)
supervised_criterion = CrossEntropyLoss()
unsupervised_criterion = MSELoss()
model = YourModel()

for x,y,is_labelled in DataLoader(al_dataset.iter_alternate(defined_length=55, p_pool=0.1), ...):
    # Let y_pred be {'reconstruction': Tensor, 'logits': Tensor}
    y_pred = model(x)
    if is_labelled:
	   loss = supervised_criterion(y,y_pred['logits'])
    else:
       loss = unsupervised_criterion(x, y_pred['reconstruction'])

```

with signature:

```python
class ActiveLearningDataset:
    def iter_alternate(self, defined_length, p_pool):
        """
        Alternate between a labelled and unlabelled datasets.
        Args:
            defined_length (int): How many steps should the iterator hold.
            p_pool (float): Probability of choosing a sample from pool.
        """
        ...
```


In this setting, we combine two iterators, the standard ActiveLearning iterator,
 but also the pool iterator.
  We will guarantee that they will alternate between the two.

## Unresolved questions

How to include this in ModelWrapper seemlessly?

## Future possibilities

### Support for label propagation.
One could support label propagation by adding a structure around the unlabelled samples.
We could keep the latest prediction and its confidence and propose it when we return items.
