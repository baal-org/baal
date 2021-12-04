from itertools import cycle
from typing import Optional, Union, Dict, Sequence, Tuple

import numpy as np
from baal.active import ActiveLearningDataset
from torch.utils.data import DataLoader


class AlternateIterator:
    """
    Create an iterator that will alternate between two dataloaders.

    Args:
        dl_1 (DataLoader): first DataLoader
        dl_2 (DataLoader): second DataLoader
        num_steps (Optional[int]): Number of steps, if None will be the sum of both length.
        p (Optional[float]): Probability of choosing dl_1 over dl_2.
            If None, will be alternate between the two.
    """

    def __init__(
        self,
        dl_1: DataLoader,
        dl_2: Optional[DataLoader],
        num_steps: Optional[int] = None,
        p: Optional[float] = None,
    ):
        self.dl_1 = dl_1
        self.dl_1_iter = cycle(dl_1)
        self.len_dl1 = len(dl_1)

        if dl_2 is not None:  # Avoid error if p=1
            self.dl_2 = dl_2
            self.dl_2_iter = cycle(dl_2)
            self.len_dl2 = len(dl_2)
        else:
            p = 1
            self.len_dl2 = 2

        self.num_steps: Optional[int] = num_steps or (self.len_dl1 + self.len_dl2)
        self.p: Optional[Tuple[float, float]] = None if p is None else (p, 1 - p)
        self._iter_idx = None

    def _make_index(self):
        if self.p is None:
            # If p is None, we just alternate.
            arr = np.array([i % 2 for i in range(self.num_steps)])
        else:
            arr = np.random.choice([0, 1], self.num_steps, p=self.p)
        return list(arr)

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        # prevent multiple __iter__ calls in _with_is_last(...) pytorch_lightning training_loop.py
        if self._iter_idx is None or len(self._iter_idx) <= 0:
            self._iter_idx = self._make_index()
        return self

    def __next__(self):
        if len(self._iter_idx) <= 0:
            raise StopIteration
        idx = self._iter_idx.pop(0)
        if idx == 0:
            return self.handle_format(next(self.dl_1_iter), idx)
        else:
            return self.handle_format(next(self.dl_2_iter), idx)

    def handle_format(self, item, idx):
        return item, idx


class SemiSupervisedIterator(AlternateIterator):
    """
    Iterator for alternating between labeled and un-labled dataloaders
    for semi-supervised learning.

    Args:
        al_dataset (ActiveLearningDataset): dataset and pool from which to load data.
        batch_size (int):  how many samples per batch to load.
        num_steps (int): number of steps before the end of the iterator.
        p (Optional[float]): probability of selecting a labeled batch.
            If None, the batches alternate.
        shuffle (Optional[bool]): set to ``True`` to have the data reshuffled at every epoch.
        num_workers (Optional[int]): how many subprocesses to use for data loading.
        drop_last (Optional[bool]): set to ``True`` to drop the last incomplete batch.
    """

    IS_LABELED_TAG = "is_labelled"

    def __init__(
        self,
        al_dataset: ActiveLearningDataset,
        batch_size: int,
        num_steps: Optional[int] = None,
        p: Optional[float] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
    ):
        self.al_dataset = al_dataset
        active_dl = DataLoader(
            al_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        pool_dl: Optional[DataLoader] = None
        if len(al_dataset.pool) > 0:
            pool_dl = DataLoader(
                al_dataset.pool,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
            )

            if num_steps is None:
                if p is None:
                    # By default num_steps if 2 times the length of active set or less
                    # This allows all the labeled data to be seen during one epoch.
                    num_steps = len(active_dl) + min(len(active_dl), len(pool_dl))
                else:
                    # Show all labeled data + unlabeled data.
                    num_steps = int(len(active_dl) + len(active_dl) * (1 - p) / p)
        else:
            p = 1

        if p == 1:
            # Allows running only supervised training.
            num_steps = len(active_dl)

        super().__init__(dl_1=active_dl, dl_2=pool_dl, num_steps=num_steps, p=p)

    def handle_format(self, item, idx):
        if isinstance(item, dict):
            item.update({self.IS_LABELED_TAG: idx == 0})
            return item
        else:
            return item, idx

    @staticmethod
    def is_labeled(batch: Union[Dict, Sequence]) -> bool:
        """
            Check if batch returned from SemiSupervisedIterator is labeled.

        Args:
            batch (Union[Dict, Sequence]): batch to check

        Returns:
            bool, if batch is labeled.

        Raises:
            ValueError if we can't process the batch type.
        """
        if isinstance(batch, dict):
            return batch[SemiSupervisedIterator.IS_LABELED_TAG]
        elif isinstance(batch, tuple):
            item, idx = batch
            return idx == 0
        else:
            raise ValueError(f"Unknown type: {type(batch)}")

    @staticmethod
    def get_batch(batch: Union[Dict, Sequence]):
        """
            Get batch without is_labeled.

        Args:
            batch (Union[Dict, Sequence]): batch and labeled indicator

        Returns:
            batch without is_labeled
        """
        if isinstance(batch, dict):
            return batch
        elif isinstance(batch, tuple):
            item, idx = batch
            return item
