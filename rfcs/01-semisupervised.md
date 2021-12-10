# Semi-supervised learning support

---
| Status        | Proposed       |
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
        Alternate between a labelled and unlabelled dataset.
        Args:
            defined_length (int): How many steps should the iterator hold.
            p_pool (float): Probability of choosing a sample from pool.
        """
        ...
```


In this setting, we combine two iterators, the standard ActiveLearning iterator,
 but also the pool iterator.
  We will guarantee that they will alternate between the two.
  
### ModelWrapper Support
 

We propose to extend `ModelWrapper`:

```python
class PoolCriterion:
    def __call__(self, data, output, target):
        raise NotImplementedError

class SSModelWrapper:
    def __init__(self, model, criterion, pool_criterion: PoolCriterion):
        ... 

    def train_on_pool_batch(self, data, target, optimizer, cuda=False):
        if cuda:
            data, target = to_cuda(data), to_cuda(target)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.pool_criterion(data, output, target)
        loss.backward()
        optimizer.step()
        self._update_metrics(output, target, loss, filter='train', is_labelled=False)
        return loss
```

Note the `is_labelled` in _update_metrics. Users will specify if a metric is for supervised or unsupervised loss.
`train_on_dataset` will be changed accordingdly to call `train_on_pool_batch` or `train_on_batch`.

## Unresolved questions

* Not sure how to make this backward compatible.

## Future possibilities

### Support for label propagation.
One could support label propagation by adding a structure around the unlabelled samples.
We could keep the latest prediction and its confidence and propose it when we return items.
