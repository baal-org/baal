from itertools import zip_longest
from typing import Tuple, Optional, Any, Union

import numpy as np

from baal.active.dataset.base import SplittedDataset


class ActiveNumpyArray(SplittedDataset):
    """
    Active dataset for numpy arrays. Useful when using sklearn.

    Args:
        dataset (Tuple[ndarray, ndarray]): [Train x, train y], The dataset.
        labelled (Union[np.ndarray, torch.Tensor]):
            An array/tensor that acts as a boolean mask which is True for every
            data point that is labelled, and False for every data point that is not
            labelled.
    """

    def __init__(
        self,
        dataset: Tuple[np.ndarray, np.ndarray],
        labelled: Optional[np.ndarray] = None,
    ) -> None:
        self.random_state = np.random.RandomState()
        if labelled is not None:
            labelled = labelled.astype(bool)
        else:
            labelled = np.zeros(len(dataset[0]), dtype=bool)
        self._dataset: Tuple[np.ndarray, np.ndarray] = dataset
        self.labelled: np.ndarray = labelled

    @property
    def pool(self):
        """Return the unlabelled portion of the dataset."""
        return self._dataset[0][~self.labelled], self._dataset[1][~self.labelled]

    @property
    def dataset(self):
        """Return the labelled portion of the dataset."""
        return self._dataset[0][self.labelled], self._dataset[1][self.labelled]

    def get_raw(self, idx: int) -> Any:
        return self._dataset[0][idx], self._dataset[1][idx]

    def __iter__(self):
        return zip(*self._dataset)

    def __getitem__(self, item):
        return self._dataset[0][item], self._dataset[1][item]

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
            self.labelled[index] = 1
