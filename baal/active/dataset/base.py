import warnings
from typing import Union, List, Optional, Any

import numpy as np
from sklearn.utils import check_random_state
from torch.utils import data as torchdata


class SplittedDataset(torchdata.Dataset):
    """Abstract class for Dataset that can be splitted.

    Args:
        labelled: An array that acts as a mask which is greater than 1 for every
            data point that is labelled, and 0 for every data point that is not
            labelled.
        random_state: Set the random seed for label_randomly().
        last_active_steps: If specified, will iterate over the last active steps
                            instead of the full dataset. Useful when doing partial finetuning.
    """

    def __init__(
        self,
        labelled,
        random_state=None,
        last_active_steps: int = -1,
    ) -> None:
        self.labelled_map = labelled
        self.random_state = check_random_state(random_state)
        if last_active_steps == 0 or last_active_steps < -1:
            raise ValueError("last_active_steps must be > 0 or -1 when disabled.")
        self.last_active_steps = last_active_steps

    def get_indices_for_active_step(self) -> List[int]:
        """Returns the indices required for the active step.

        Returns the indices of the labelled items. Also takes into account self.last_active_step.

        Returns:
            List of the selected indices for training.
        """
        if self.last_active_steps == -1:
            min_labelled_step = 0
        else:
            min_labelled_step = max(0, self.current_al_step - self.last_active_steps)

        # we need to work with lists since arrow dataset is not compatible with np.int types!
        indices = [indx for indx, val in enumerate(self.labelled_map) if val > min_labelled_step]
        return indices

    def is_labelled(self, idx: int) -> bool:
        """Check if a datapoint is labelled."""
        return bool(self.labelled[idx].item() == 1)

    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return len(self.get_indices_for_active_step())

    def __getitem__(self, index):
        raise NotImplementedError

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

    @property
    def _labelled(self):
        warnings.warn(
            "_labelled as been renamed labelled. Please update your script.", DeprecationWarning
        )
        return self.labelled

    @property
    def current_al_step(self) -> int:
        """Get the current active learning step."""
        return int(self.labelled_map.max())

    @property
    def labelled(self):
        """An array that acts as a boolean mask which is True for every
        data point that is labelled, and False for every data point that is not
        labelled."""
        return self.labelled_map.astype(bool)

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
