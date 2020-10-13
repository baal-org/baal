# Coreset API


Coresets (or core-sets) are methods that tries to select of subset of a dataset that will best represent the true distribution.

## Proposition

Coresets are similar to heuristics, but instead of selecting data based on uncertainty, they would use features.

Gathering features is easy, we can use a `forward_hook`.

**API**
```python
Array = np.ndarray
Indices = List[int]

class BaseCoreset:
    def get_ranks(features: Array, logits: Array, top_k:int =None) -> Indices:
        # K-Means or something
        pass
```


### Concerns

* Speed
    * Clustering can take a long time.
* Memory
    * Keeping features in memory will be expensive for most users.

### MVP

We need to reproduce [1].

## References

[1] Active Learning for Convolutional Neural Networks: A Core-Set Approach, arxiv.org/abs/1708.00489



[2] Introduction to Coresets: Accurate Coresets arxiv.org/abs/1910.08707
