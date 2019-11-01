import unittest

import numpy as np
import pytest
import torch
from sklearn.datasets import load_iris
from torch.utils.data import Dataset
from torchvision.transforms import Lambda

from baal.active import ActiveLearningDataset
from baal.active.dataset import ActiveNumpyArray


class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        pass

    def __len__(self):
        return 100

    def __getitem__(self, item):
        feature = item
        if self.transform:
            feature = self.transform(item)
        return (feature, item)


class ActiveDatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = ActiveLearningDataset(MyDataset(),
                                             make_unlabelled=lambda x: (x[0], -1))

    def test_len(self):
        assert len(self.dataset) == 0
        assert self.dataset.n_unlabelled == 100
        assert len(self.dataset.pool) == 100
        self.dataset.label(0)
        assert len(self.dataset) == self.dataset.n_labelled == 1
        assert self.dataset.n_unlabelled == 99
        assert len(self.dataset.pool) == 99
        self.dataset.label(list(range(99)))
        assert len(self.dataset) == 100
        assert self.dataset.n_unlabelled == 0
        assert len(self.dataset.pool) == 0

        dummy_dataset = ActiveLearningDataset(MyDataset(), labelled=self.dataset._labelled,
                                              make_unlabelled=lambda x: (x[0], -1))
        assert len(dummy_dataset) == len(self.dataset)
        assert len(dummy_dataset.pool) == len(self.dataset.pool)

        dummy_lbl = torch.from_numpy(self.dataset._labelled.astype(np.float32))
        dummy_dataset = ActiveLearningDataset(MyDataset(), labelled=dummy_lbl,
                                              make_unlabelled=lambda x: (x[0], -1))
        assert len(dummy_dataset) == len(self.dataset)
        assert len(dummy_dataset.pool) == len(self.dataset.pool)

    def test_pool(self):
        self.dataset._dataset.label = unittest.mock.MagicMock()
        labels_initial = self.dataset.n_labelled
        self.dataset.can_label = False
        self.dataset.label(0, value=np.arange(1, 10))
        self.dataset._dataset.label.assert_not_called()
        labels_next_1 = self.dataset.n_labelled
        assert labels_next_1 == labels_initial + 1
        self.dataset.can_label = True
        self.dataset.label(np.arange(0, 9))
        self.dataset._dataset.label.assert_not_called()
        labels_next_2 = self.dataset.n_labelled
        assert labels_next_1 == labels_next_2
        self.dataset.label(np.arange(0, 9), value=np.arange(1, 10))
        assert self.dataset._dataset.label.called_once_with(np.arange(1, 10))
        # cleanup
        del self.dataset._dataset.label
        self.dataset.can_label = False

        pool = self.dataset.pool
        assert np.equal([i for i in pool], [(i, -1) for i in np.arange(2, 100)]).all()
        assert np.equal([i for i in self.dataset], [(i, i) for i in np.arange(2)]).all()

    def test_get_raw(self):
        # check that get_raw returns the same thing regardless of labelling
        # status
        i_1 = self.dataset.get_raw(5)
        self.dataset.label(5)
        i_2 = self.dataset.get_raw(5)
        assert i_1 == i_2

    def test_types(self):
        self.dataset.label_randomly(2)
        assert self.dataset._pool_to_oracle_index(1) == self.dataset._pool_to_oracle_index([1])
        assert self.dataset._oracle_to_pool_index(1) == self.dataset._oracle_to_pool_index([1])

    def test_state_dict(self):
        state_dict_1 = self.dataset.state_dict()
        assert np.equal(state_dict_1["labelled"], np.full((100,), False)).all()

        self.dataset.label(0)
        assert np.equal(
            state_dict_1["labelled"],
            np.concatenate((np.array([True]), np.full((99,), False)))
        ).all()

    def test_load_state_dict(self):
        dataset_1 = ActiveLearningDataset(MyDataset(), random_state=50)
        dataset_1.label_randomly(10)
        state_dict1 = dataset_1.state_dict()

        dataset_2 = ActiveLearningDataset(MyDataset(), random_state=None)
        assert dataset_2.n_labelled == 0

        dataset_2.load_state_dict(state_dict1)
        assert dataset_2.n_labelled == 10

        # test if the second lable_randomly call have same behaviour
        dataset_1.label_randomly(5)
        dataset_2.label_randomly(5)

        assert np.allclose(dataset_1._labelled, dataset_2._labelled)

    def test_transform(self):
        train_transform = Lambda(lambda k: 1)
        test_transform = Lambda(lambda k: 0)
        dataset = ActiveLearningDataset(MyDataset(train_transform),
                                        pool_specifics={'transform': test_transform},
                                        make_unlabelled=lambda x: (x[0], -1))
        dataset.label(np.arange(10))
        pool = dataset.pool
        assert np.equal([i for i in pool], [(0, -1) for i in np.arange(10, 100)]).all()
        assert np.equal([i for i in dataset], [(1, i) for i in np.arange(10)]).all()

        with pytest.warns(DeprecationWarning) as e:
            ActiveLearningDataset(MyDataset(train_transform), eval_transform=train_transform)
        assert len(e) == 1

        with pytest.raises(ValueError) as e:
            ActiveLearningDataset(MyDataset(train_transform), pool_specifics={'whatever': 123}).pool

    def test_random(self):
        self.dataset.label_randomly(50)
        assert len(self.dataset) == 50
        assert len(self.dataset.pool) == 50

    def test_random_state(self):
        seed = None
        dataset_1 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_1.label_randomly(10)
        dataset_2 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_2.label_randomly(10)
        assert not np.allclose(dataset_1._labelled, dataset_2._labelled)

        seed = 50
        dataset_1 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_1.label_randomly(10)
        dataset_2 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_2.label_randomly(10)
        assert np.allclose(dataset_1._labelled, dataset_2._labelled)

        seed = np.random.RandomState(50)
        dataset_1 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_1.label_randomly(10)
        dataset_2 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_2.label_randomly(10)
        assert not np.allclose(dataset_1._labelled, dataset_2._labelled)

    def test_label_randomly_full(self):
        dataset_1 = ActiveLearningDataset(MyDataset())
        dataset_1.label_randomly(99)
        assert dataset_1.n_unlabelled == 1
        assert len(dataset_1.pool) == 1
        dataset_1.label_randomly(1)
        assert dataset_1.n_unlabelled == 0
        assert dataset_1.n_labelled == 100


def test_numpydataset():
    x, y = load_iris(return_X_y=True)
    init_len = len(x)
    dataset = ActiveNumpyArray((x, y))
    assert len(dataset) == 0 == dataset.n_labelled
    assert dataset.n_unlabelled == init_len

    dataset.label_randomly(10)
    assert len(dataset) == 10 == dataset.n_labelled
    assert dataset.n_unlabelled == init_len - 10

    xi, yi = dataset.dataset
    assert len(xi) == 10

    xp, yp = dataset.pool
    assert len(xp) == init_len - 10

    dataset.label(list(range(10)))
    assert len(dataset) == 20

    l = np.array([1] * 10 + [0] * (init_len - 10))
    dataset = ActiveNumpyArray((x, y), labelled=l)
    assert len(dataset) == 10

    assert [a == b for a, b in zip(dataset.get_raw(-1), (x[-1], y[-1]))]
    assert [a == b for a, b in zip(dataset.get_raw(0), (x[0], y[0]))]

    assert (next(iter(dataset))[0] == dataset[0][0]).all()


if __name__ == '__main__':
    pytest.main()
