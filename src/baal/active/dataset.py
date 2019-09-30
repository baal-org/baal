from copy import copy
from itertools import zip_longest
from typing import Union, Optional, Callable, Tuple, List, Any

import numpy as np
import torch
import torch.utils.data as torchdata


class ActiveLearningDataset(torchdata.Dataset):
    """A dataset that allows for active learning.

    Args:
        dataset (torch.data.Dataset): The baseline dataset.
        eval_transform (Optional(Callable)): transformations to call on the evaluation dataset.
        labelled (Union[np.ndarray, torch.Tensor]):
            An array/tensor that acts as a boolean mask which is True for every
            data point that is labelled, and False for every data point that is not
            labelled.
        make_unlabelled (Callable): the function that returns an
            unlabelled version of a datum so that it can still be used in the DataLoader.
    """

    def __init__(
        self,
        dataset: torchdata.Dataset,
        eval_transform: Optional[Callable] = None,
        labelled: Union[np.ndarray, torch.Tensor] = None,
        make_unlabelled: Callable = lambda x: x,
    ) -> None:
        self._dataset = dataset
        if labelled is not None:
            if isinstance(labelled, torch.Tensor):
                labelled = labelled.numpy()
            self._labelled = labelled.astype(np.bool)
        else:
            self._labelled = np.zeros(len(self._dataset), dtype=np.bool)
        if eval_transform is not None:
            self.eval_transform = eval_transform
        else:
            self.eval_transform = getattr(self._dataset, 'transform', None)

        self.make_unlabelled = make_unlabelled
        # For example, FileDataset has a method 'label'. This is useful when we're in prod.
        self.can_label = hasattr(self._dataset, 'label')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Return stuff from the original dataset."""
        return self._dataset[self._labelled_to_oracle_index(index)]

    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return self._labelled.sum()

    class ActiveIter():
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
    def n_unlabelled(self):
        """The number of unlabelled data points."""
        return (~self._labelled).sum()

    @property
    def n_labelled(self):
        """The number of labelled data points."""
        return self._labelled.sum()

    @property
    def pool(self) -> torchdata.Dataset:
        """Returns a new Dataset made from unlabelled samples"""
        pool_dataset = copy(self._dataset)
        # TODO Handle target transform as well.
        if hasattr(pool_dataset, 'transform') and self.eval_transform is not None:
            pool_dataset.transform = self.eval_transform

        pool_dataset = torchdata.Subset(pool_dataset,
                                        (~self._labelled).nonzero()[0].squeeze())
        ald = ActiveLearningPool(pool_dataset, make_unlabelled=self.make_unlabelled)
        return ald

    """ This returns one or zero, if it is labelled or not, no index is returned.
    """

    def _labelled_to_oracle_index(self, index: int) -> int:
        return self._labelled.nonzero()[0][index].squeeze().item()

    def _pool_to_oracle_index(self, index: Union[int, List[int]]) -> List[int]:
        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index]

        lbl_nz = (~self._labelled).nonzero()[0]
        return [int(lbl_nz[idx].squeeze().item()) for idx in index]

    def _oracle_to_pool_index(self, index: Union[int, List[int]]) -> List[int]:
        if isinstance(index, int):
            index = [index]

        # Pool indices are the unlabelled, starts at 0
        lbl_cs = np.cumsum(~self._labelled) - 1
        return [int(lbl_cs[idx].squeeze().item()) for idx in index]

    def label(self, index: Union[list, int], value: Optional[Any] = None) -> None:
        """
        Label data points.
        The index should be relative to the pool, not the overall data.

        Args:
            index (Union[list,int]): one or many indices to label.
            value (Optional[Any]): The label value. If not provided, no modification
                                    to the underlying dataset is done.
        """
        if isinstance(index, int):
            index = [index]
        if not isinstance(value, (list, tuple)):
            value = [value]
        indexes = self._pool_to_oracle_index(index)
        for index, val in zip_longest(indexes, value, fillvalue=None):
            if self.can_label and val is not None:
                self._dataset.label(index, val)
            self._labelled[index] = 1

    def label_randomly(self, n: int = 1) -> None:
        """
        Label `n` data-points randomly.

        Args:
            n (int): number of samples to label.
        """
        for i in range(n):
            """Making multiple call to self.n_unlabelled is eneficient, but
            self.label changes the available length and it may lead to
            IndexError if not done this way."""
            self.label(np.random.choice(self.n_unlabelled, 1).item())

    def reset_labeled(self):
        """Reset the label map."""
        self._labelled = np.zeros(len(self._dataset), dtype=np.bool)

    def is_labelled(self, idx: int) -> bool:
        """Check if a datapoint is labelled"""
        return self._labelled[idx] == 1

    def get_raw(self, idx: int) -> None:
        """Get a datapoint from the underlying dataset."""
        return self._dataset[idx]

    def state_dict(self):
        """Return the state_dict, ie. the labelled map."""
        return {'labeled': self._labelled}


class ActiveLearningPool(torchdata.Dataset):
    """ A dataset that represents the unlabelled pool for active learning.

    Args:
        dataset (Dataset): A Dataset object providing unlabeled sample.
        make_unlabelled (Callable): the function that returns an
            unlabelled version of a datum so that it can still be used in the DataLoader.

    """

    def __init__(self, dataset: torchdata.Dataset, make_unlabelled: Callable = lambda x: x) -> None:
        self._dataset = dataset
        self.make_unlabelled = make_unlabelled

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        # this datum is marked as unlabelled, so clear the label
        return self.make_unlabelled(self._dataset[index])

    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return len(self._dataset)
