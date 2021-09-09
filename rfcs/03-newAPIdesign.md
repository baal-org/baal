# Semi-supervised learning support

---
| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | 3|
| **Author(s)** | @parmidaatg |
| **Sponsor**   | None                 |
| **Updated**   | 2021-09-09                                           |


---

## Summary
The new design proposition tries to incorporate an easier integration of different active learning algorithms. This project started on the bases of using
MCDropout as the core of bayesian approximation in the model. However, as we move forward, we would like to make it easier for the user to use theirown approach
for approximating the model weight distribution. In order to prevent too much complexity and easier mix and match between the existing algorithms which is the main
premise of this repository, we can replace the `predict` functionality in all the supported training pipelines (currently, ModelWrapper, TransformerWrapper, SSLWrapper and PLWrapper)
with a custom defined function which calculates the required distribution for a given heuristic to use. In the case of our current supported heuristics,
the output is distribution of model prediction for almost all heuristics and embedding gradients for BADGE.

## Guide-level explanation
 
The ActiveLoop signature would not change but all the existing supported wrappers would change similar to below example on ModelWrapper.
```python
from pydantic import BaseClass
from baal.active.al_distibution_calculator import *
class AL_Distibution(BaseClass):
    'mcd': get_mcd_predictions
    'emb_grads': get_embedding_grads

class ModelWrapper(self,
                   model,
                   criterion,
                   get_distribution: str,
                   replicate_in_memory=True,
                   **kwargs):
...
    def get_distribution(self,
                         dataset: Dataset,
                         batch_size: int,
                         iterations: int,
                         use_cuda: bool,
                         workers: int = 4,
                         collate_fn: Optional[Callable] = None,
                         half=False,
                         verbose=True):
        return AL_Distibution[get_distribution](dataset: Dataset,
                                                batch_size: int,
                                                use_cuda: bool,
                                                workers: int = 4,
                                                collate_fn: Optional[Callable] = None,
                                                half=False,
                                                verbose=True,
                                                **kwargs)           
```


Note that the input variables specific to the method will be provide via `kwargs` (ex: `iteration` for `mcd`)

