import warnings
from copy import copy
from itertools import zip_longest
from typing import Union, Optional, Callable, Tuple, List, Any

import numpy as np
import torch
import torch.utils.data as torchdata
from sklearn.utils import check_random_state


def _identity(x):
    return x


class ActiveLearningDataset(torchdata.Dataset):
    """A dataset that allows for active learning.

    Args:
        dataset (torch.data.Dataset): The baseline dataset.
        labelled (Union[np.ndarray, torch.Tensor]):
            An array/tensor that acts as a boolean mask which is True for every
            data point that is labelled, and False for every data point that is not
            labelled.
        make_unlabelled (Callable): The function that returns an
            unlabelled version of a datum so that it can still be used in the DataLoader.
        random_state (None, int, RandomState): Set the random seed for label_randomly().
        pool_specifics (Optional[Dict]): Attributes to set when creating the pool.
                                         Useful to remove data augmentation.
    """

    def __init__(self, dataset: torchdata.Dataset, labelled: Union[np.ndarray, torch.Tensor] = None,
                 make_unlabelled: Callable = _identity, random_state=None,
                 pool_specifics: Optional[dict] = None) -> None:
        self._dataset = dataset
        if labelled is not None:
            if isinstance(labelled, torch.Tensor):
                labelled = labelled.numpy()
            self.labelled = labelled.astype(bool)
        else:
            self.labelled = np.zeros(len(self._dataset), dtype=bool)

        if pool_specifics is None:
            pool_specifics = {}
        self.pool_specifics = pool_specifics

        self.make_unlabelled = make_unlabelled
        # For example, FileDataset has a method 'label'. This is useful when we're in prod.
        self.can_label = self.check_dataset_can_label()

        self.random_state = check_random_state(random_state)

    @property
    def _labelled(self):
        warnings.warn("_labelled as been renamed labelled. Please update your script.",
                      DeprecationWarning)
        return self.labelled

    def check_dataset_can_label(self):
        """Check if a dataset can be labelled.

        Returns:
            Whether the dataset's label can be modified or not.

        Notes:
            To be labelled, a dataset needs a method `label`
            with definition: `label(self, idx, value)` where `value`
            is the label for indice `idx`.
        """
        has_label_attr = hasattr(self._dataset, 'label')
        if has_label_attr:
            if callable(self._dataset.label):
                return True
            else:
                warnings.warn('Dataset has an attribute `label`, but it is not callable.'
                              'The Dataset will not be labelled with new labels.',
                              UserWarning)
        return False

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Return stuff from the original dataset."""
        return self._dataset[self._labelled_to_oracle_index(index)]

    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return self.labelled.sum()

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
        return (~self.labelled).sum()

    @property
    def n_labelled(self):
        """The number of labelled data points."""
        return self.labelled.sum()

    @property
    def pool(self) -> torchdata.Dataset:
        """Returns a new Dataset made from unlabelled samples.

        Raises:
            ValueError if a pool specific attribute cannot be set.
        """
        pool_dataset = copy(self._dataset)

        for attr, new_val in self.pool_specifics.items():
            if hasattr(pool_dataset, attr):
                setattr(pool_dataset, attr, new_val)
            else:
                raise ValueError(f"{pool_dataset} doesn't have {attr}")

        pool_dataset = torchdata.Subset(pool_dataset,
                                        (~self.labelled).nonzero()[0].reshape([-1]))
        ald = ActiveLearningPool(pool_dataset, make_unlabelled=self.make_unlabelled)
        return ald

    """ This returns one or zero, if it is labelled or not, no index is returned.
    """

    def _labelled_to_oracle_index(self, index: int) -> int:
        return self.labelled.nonzero()[0][index].squeeze().item()

    def _pool_to_oracle_index(self, index: Union[int, List[int]]) -> List[int]:
        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index]

        lbl_nz = (~self.labelled).nonzero()[0]
        return [int(lbl_nz[idx].squeeze().item()) for idx in index]

    def _oracle_to_pool_index(self, index: Union[int, List[int]]) -> List[int]:
        if isinstance(index, int):
            index = [index]

        # Pool indices are the unlabelled, starts at 0
        lbl_cs = np.cumsum(~self.labelled) - 1
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
                self.labelled[index] = 1
            elif self.can_label and val is None:
                warnings.warn("""The dataset is able to label data, but no label was provided.
                                 The dataset will be unchanged from this action!
                                 If this is a research setting, please set the
                                  `ActiveLearningDataset.can_label` to `False`.
                                  """, UserWarning)
            elif not self.can_label:
                self.labelled[index] = 1
                if val is not None:
                    warnings.warn(
                        "We will consider the original label of this datasample : {}, {}.".format(
                            self._dataset[index][0], self._dataset[index][1]), UserWarning)

    def label_randomly(self, n: int = 1) -> None:
        """
        Label `n` data-points randomly.

        Args:
            n (int): Number of samples to label.
        """
        for i in range(n):
            """Making multiple call to self.n_unlabelled is inefficient, but
            self.label changes the available length and it may lead to
            IndexError if not done this way."""
            self.label(self.random_state.choice(self.n_unlabelled, 1).item())

    def reset_labeled(self):
        """Reset the label map."""
        self.labelled = np.zeros(len(self._dataset), dtype=np.bool)

    def is_labelled(self, idx: int) -> bool:
        """Check if a datapoint is labelled."""
        return self.labelled[idx] == 1

    def get_raw(self, idx: int) -> None:
        """Get a datapoint from the underlying dataset."""
        return self._dataset[idx]

    def state_dict(self):
        """Return the state_dict, ie. the labelled map and random_state."""
        return {'labelled': self.labelled,
                'random_state': self.random_state}

    def load_state_dict(self, state_dict):
        """Load the labelled map and random_state with give state_dict."""
        self.labelled = state_dict['labelled']
        self.random_state = state_dict['random_state']


class ActiveLearningPool(torchdata.Dataset):
    """ A dataset that represents the unlabelled pool for active learning.

    Args:
        dataset (Dataset): A Dataset object providing unlabelled sample.
        make_unlabelled (Callable): The function that returns an
            unlabelled version of a datum so that it can still be used in the DataLoader.

    """

    def __init__(self, dataset: torchdata.Dataset, make_unlabelled: Callable = _identity) -> None:
        self._dataset = dataset
        self.make_unlabelled = make_unlabelled

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        # This datum is marked as unlabelled, so clear the label.
        return self.make_unlabelled(self._dataset[index])

    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return len(self._dataset)


class ActiveNumpyArray(ActiveLearningDataset):
    """
    Active dataset for numpy arrays. Useful when using sklearn.

    Args:
        dataset (Tuple[ndarray, ndarray]): [Train x, train y], The dataset.
        labelled (Union[np.ndarray, torch.Tensor]):
            An array/tensor that acts as a boolean mask which is True for every
            data point that is labelled, and False for every data point that is not
            labelled.
    """

    def __init__(self, dataset: Tuple[np.ndarray, np.ndarray],
                 labelled: Union[np.ndarray, torch.Tensor] = None) -> None:

        if labelled is not None:
            if isinstance(labelled, torch.Tensor):
                labelled = labelled.numpy()
            labelled = labelled.astype(bool)
        else:
            labelled = np.zeros(len(dataset[0]), dtype=bool)
        super().__init__(dataset, labelled=labelled)

    @property
    def pool(self):
        """Return the unlabelled portion of the dataset."""
        return self._dataset[0][~self.labelled], self._dataset[1][~self.labelled]

    @property
    def dataset(self):
        """Return the labelled portion of the dataset."""
        return self._dataset[0][self.labelled], self._dataset[1][self.labelled]

    def get_raw(self, idx: int) -> None:
        return self._dataset[0][idx], self._dataset[1][idx]

    def __iter__(self):
        return zip(*self._dataset)
