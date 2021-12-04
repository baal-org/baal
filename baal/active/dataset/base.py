from typing import Callable, Union, List, Optional, Any

import numpy as np
from torch.utils import data as torchdata


class SplittedDataset(torchdata.Dataset):
    """Abstract class for Dataset that can be splitted."""

    labelled: np.ndarray
    random_state: np.random.RandomState

    def is_labelled(self, idx: int) -> bool:
        """Check if a datapoint is labelled."""
        return bool(self.labelled[idx].item() == 1)

    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return int(self.labelled.sum())

    @property
    def n_unlabelled(self):
        """The number of unlabelled data points."""
        return (~self.labelled).sum()

    @property
    def n_labelled(self):
        """The number of labelled data points."""
        return self.labelled.sum()

    def label(self, index: Union[list, int], value: Optional[Any] = None) -> None:
        """
        Label data points.
        The index should be relative to the pool, not the overall data.

        Args:
            index: one or many indices to label.
            value: The label value. If not provided, no modification
                                    to the underlying dataset is done.
        """
        raise NotImplementedError

    def label_randomly(self, n: int = 1) -> None:
        """
        Label `n` data-points randomly.

        Args:
            n (int): Number of samples to label.
        """
        self.label(self.random_state.choice(self.n_unlabelled, n, replace=False).tolist())

    def _labelled_to_oracle_index(self, index: int) -> int:
        return int(self.labelled.nonzero()[0][index].squeeze().item())

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
