import warnings
from copy import deepcopy
from itertools import zip_longest
from typing import Union, Optional, Callable, Any, Dict, List, Sequence, Mapping

import numpy as np
import torch
import torch.utils.data as torchdata
from torch import Tensor

from baal.active.dataset.base import SplittedDataset, Dataset
from baal.utils.equality import deep_check

STOCHASTIC_POOL_WARNING = """
It seems that data augmentation is not disabled when iterating on the pool.
You can disable it by overriding attributes using `pool_specifics` 
when instantiating ActiveLearningDataset.
Example:
```
from torchvision.transforms import *
train_transform = Compose([Resize((224, 224)), RandomHorizontalFlip(),
                            RandomRotation(30), ToTensor()])
test_transform = Compose([Resize((224, 224)),ToTensor()])
dataset = CIFAR10(..., transform=train_transform)

al_dataset = ActiveLearningDataset(dataset,
                                    pool_specifics={'transform': test_transform})
```   
"""


def _identity(x):
    return x


class ActiveLearningDataset(SplittedDataset):
    """A dataset that allows for active learning.

    Args:
        dataset: The baseline dataset.
        labelled: An array that acts as a mask which is greater than 1 for every
            data point that is labelled, and 0 for every data point that is not
            labelled.
        make_unlabelled: The function that returns an
            unlabelled version of a datum so that it can still be used in the DataLoader.
        random_state: Set the random seed for label_randomly().
        pool_specifics: Attributes to set when creating the pool.
                                         Useful to remove data augmentation.
        last_active_steps: If specified, will iterate over the last active steps
                            instead of the full dataset. Useful when doing partial finetuning.
    """

    def __init__(
        self,
        dataset: Dataset,
        labelled: Optional[np.ndarray] = None,
        make_unlabelled: Callable = _identity,
        random_state=None,
        pool_specifics: Optional[dict] = None,
        last_active_steps: int = -1,
    ) -> None:
        self._dataset = dataset

        # The labelled_map keeps track of the step at which an item as been labelled.
        if labelled is not None:
            labelled_map: np.ndarray = labelled.astype(int)
        else:
            labelled_map = np.zeros(len(self._dataset), dtype=int)

        if pool_specifics is None:
            pool_specifics = {}
        self.pool_specifics: Dict[str, Any] = pool_specifics

        self.make_unlabelled = make_unlabelled
        # For example, FileDataset has a method 'label'. This is useful when we're in prod.
        self.can_label = self.check_dataset_can_label()
        super().__init__(
            labelled=labelled_map, random_state=random_state, last_active_steps=last_active_steps
        )
        self._warn_if_pool_stochastic()

    def check_dataset_can_label(self):
        """Check if a dataset can be labelled.

        Returns:
            Whether the dataset's label can be modified or not.

        Notes:
            To be labelled, a dataset needs a method `label`
            with definition: `label(self, idx, value)` where `value`
            is the label for indice `idx`.
        """
        has_label_attr = getattr(self._dataset, "label", None)
        if has_label_attr:
            if callable(has_label_attr):
                return True
            else:
                warnings.warn(
                    "Dataset has an attribute `label`, but it is not callable."
                    "The Dataset will not be labelled with new labels.",
                    UserWarning,
                )
        return False

    def __getitem__(self, index: int) -> Any:
        """Return items from the original dataset based on the labelled index."""
        index = self.get_indices_for_active_step()[index]
        return self._dataset[index]

    class ActiveIter:
        """Iterator over an ActiveLearningDataset."""

        def __init__(self, aldataset):
            self.i = 0
            self.aldataset = aldataset

        def __len__(self):
            return len(self.aldataset)

        def __next__(self):
            if self.i >= len(self):
                raise StopIteration

            n = self.aldataset[self.i]
            self.i = self.i + 1
            return n

    def __iter__(self):
        return self.ActiveIter(self)

    @property
    def pool(self) -> "ActiveLearningPool":
        """Returns a new Dataset made from unlabelled samples.

        Raises:
            ValueError if a pool specific attribute cannot be set.
        """
        current_dataset = deepcopy(self._dataset)

        for attr, new_val in self.pool_specifics.items():
            if hasattr(current_dataset, attr):
                setattr(current_dataset, attr, new_val)
            else:
                raise ValueError(f"{current_dataset} doesn't have {attr}")

        pool_dataset: torchdata.Subset = torchdata.Subset(
            current_dataset, (~self.labelled).nonzero()[0].reshape([-1]).tolist()
        )
        ald = ActiveLearningPool(pool_dataset, make_unlabelled=self.make_unlabelled)
        return ald

    def label(self, index: Union[list, int], value: Optional[Any] = None) -> None:
        """
        Label data points.
        The index should be relative to the pool, not the overall data.

        Args:
            index: one or many indices to label.
            value: The label value. If not provided, no modification
                                    to the underlying dataset is done.

        Raises:
            ValueError if the indices do not match the values or
             if no `value` is provided and `can_label` is True.
        """
        if isinstance(index, int):
            # We were provided only the index, we make a list.
            index_lst = [index]
            value_lst: List[Any] = [value]
        else:
            index_lst = index
            if value is None:
                value_lst = [value]
            else:
                value_lst = value

        if value_lst[0] is not None and len(index_lst) != len(value_lst):
            raise ValueError(
                "Expected `index` and `value` to be of same length when `value` is provided."
                f"Got index={len(index_lst)} and value={len(value_lst)}"
            )
        indexes = self._pool_to_oracle_index(index_lst)
        active_step = self.current_al_step + 1
        for idx, val in zip_longest(indexes, value_lst, fillvalue=None):
            if self.can_label and val is not None:
                self._dataset.label(idx, val)  # type: ignore
                self.labelled_map[idx] = active_step
            elif self.can_label and val is None:
                raise ValueError(
                    """The dataset is able to label data, but no label was provided.
                                 If this is a research setting, please set the
                                  `ActiveLearningDataset.can_label` to `False`.
                                  """
                )
            else:
                # Regular research usecase.
                self.labelled_map[idx] = active_step
                if val is not None:
                    warnings.warn(
                        "We will consider the original label of this datasample : {}, {}.".format(
                            self._dataset[idx][0], self._dataset[idx][1]
                        ),
                        UserWarning,
                    )

    def reset_labelled(self):
        """Reset the label map."""
        self.labelled_map = np.zeros(len(self._dataset), dtype=np.bool)

    def get_raw(self, idx: int) -> Any:
        """Get a datapoint from the underlying dataset."""
        return self._dataset[idx]

    def state_dict(self) -> Dict:
        """Return the state_dict, ie. the labelled map and random_state."""
        return {"labelled": self.labelled_map, "random_state": self.random_state}

    def load_state_dict(self, state_dict):
        """Load the labelled map and random_state with give state_dict."""
        self.labelled_map = state_dict["labelled"]
        self.random_state = state_dict["random_state"]

    def _warn_if_pool_stochastic(self):
        pool = self.pool
        if len(pool) > 0 and not deep_check(pool[0], pool[0]):
            warnings.warn(
                STOCHASTIC_POOL_WARNING,
                UserWarning,
            )


class ActiveLearningPool(torchdata.Dataset):
    """A dataset that represents the unlabelled pool for active learning.

    Args:
        dataset (Dataset): A Dataset object providing unlabelled sample.
        make_unlabelled (Callable): The function that returns an
            unlabelled version of a datum so that it can still be used in the DataLoader.

    """

    def __init__(self, dataset: torchdata.Subset, make_unlabelled: Callable = _identity) -> None:
        self._dataset: torchdata.Subset = dataset
        self.make_unlabelled = make_unlabelled

    def __getitem__(self, index: int) -> Any:
        # This datum is marked as unlabelled, so clear the label.
        return self.make_unlabelled(self._dataset[index])

    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return len(self._dataset)
