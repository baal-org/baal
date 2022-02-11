import pickle
import unittest
import unittest.mock
import warnings

import datasets as HFdata
import numpy as np
import pytest
import torch.utils.data as torchdata
from torchvision.transforms import Lambda

from baal.active import ActiveLearningDataset


class MyDataset(torchdata.Dataset):
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
        self.dataset = ActiveLearningDataset(MyDataset(), make_unlabelled=lambda x: (x[0], -1))

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

        dummy_dataset = ActiveLearningDataset(MyDataset(), labelled=self.dataset.labelled,
                                              make_unlabelled=lambda x: (x[0], -1))
        assert len(dummy_dataset) == len(self.dataset)
        assert len(dummy_dataset.pool) == len(self.dataset.pool)

        dummy_lbl = self.dataset.labelled.astype(np.float32)
        dummy_dataset = ActiveLearningDataset(MyDataset(), labelled=dummy_lbl, make_unlabelled=lambda x: (x[0], -1))
        assert len(dummy_dataset) == len(self.dataset)
        assert len(dummy_dataset.pool) == len(self.dataset.pool)

    def test_pool(self):
        self.dataset._dataset.label = unittest.mock.MagicMock()
        labels_initial = self.dataset.n_labelled

        # Test that we can label without value for research
        self.dataset.can_label = False
        self.dataset.label(0, value=np.arange(1, 10))
        self.dataset._dataset.label.assert_not_called()
        labels_next_1 = self.dataset.n_labelled
        assert labels_next_1 == labels_initial + 1

        # Test that a warning is raised if we dont supply a label.
        self.dataset.can_label = True
        with pytest.raises(ValueError, match="The dataset is able to label data, but no label was provided.") as w:
            self.dataset.label(np.arange(0, 9))
        self.dataset._dataset.label.assert_not_called()
        assert self.dataset.n_labelled == labels_next_1

        with pytest.raises(ValueError, match="same length"):
            self.dataset.label(np.arange(0, 9), value=np.arange(0, 10))

        # cleanup
        del self.dataset._dataset.label
        self.dataset.can_label = False

        pool = self.dataset.pool
        assert np.equal([i for i in pool], [(i, -1) for i in np.arange(labels_next_1, 100)]).all()
        assert np.equal([i for i in self.dataset], [(i, i) for i in np.arange(labels_next_1)]).all()

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

        assert np.allclose(dataset_1.labelled, dataset_2.labelled)

    def test_transform(self):
        train_transform = Lambda(lambda k: 1)
        test_transform = Lambda(lambda k: 0)
        dataset = ActiveLearningDataset(MyDataset(train_transform), make_unlabelled=lambda x: (x[0], -1),
                                        pool_specifics={'transform': test_transform})
        dataset.label(np.arange(10))
        pool = dataset.pool
        assert np.equal([i for i in pool], [(0, -1) for i in np.arange(10, 100)]).all()
        assert np.equal([i for i in dataset], [(1, i) for i in np.arange(10)]).all()

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
        assert not np.allclose(dataset_1.labelled, dataset_2.labelled)

        seed = 50
        dataset_1 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_1.label_randomly(10)
        dataset_2 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_2.label_randomly(10)
        assert np.allclose(dataset_1.labelled, dataset_2.labelled)

        seed = np.random.RandomState(50)
        dataset_1 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_1.label_randomly(10)
        dataset_2 = ActiveLearningDataset(MyDataset(), random_state=seed)
        dataset_2.label_randomly(10)
        assert not np.allclose(dataset_1.labelled, dataset_2.labelled)

    def test_label_randomly_full(self):
        dataset_1 = ActiveLearningDataset(MyDataset())
        dataset_1.label_randomly(99)
        assert dataset_1.n_unlabelled == 1
        assert len(dataset_1.pool) == 1
        dataset_1.label_randomly(1)
        assert dataset_1.n_unlabelled == 0
        assert dataset_1.n_labelled == 100


def test_arrowds():
    dataset = HFdata.load_dataset('glue', 'sst2')['test']
    dataset = ActiveLearningDataset(dataset)
    dataset.label(np.arange(10))
    assert len(dataset) == 10
    assert len(dataset.pool) == 1811
    data = dataset.pool[0]
    assert [k in ['idx', 'label', 'sentence'] for k, v in data.items()]


def test_pickable():
    dataset_1 = ActiveLearningDataset(MyDataset())
    l = len(dataset_1.pool)
    dataset_2 = pickle.loads(pickle.dumps(dataset_1))
    l2 = len(dataset_2.pool)
    assert l == l2


def test_warning_raised_on_label():
    class DS(torchdata.Dataset):
        def __init__(self):
            self.x = [1, 2, 3]
            self.label = [1, 1, 1]

        def __len__(self):
            return len(self.x)

        def __getitem__(self, item):
            return self.x[item], self.y[item]

    with warnings.catch_warnings(record=True) as w:
        al = ActiveLearningDataset(DS())
        assert not al.can_label
        assert len(w) == 1
        assert "label" in str(w[-1].message)


def test_warning_raised_on_deprecation():
    with warnings.catch_warnings(record=True) as w:
        al = ActiveLearningDataset(MyDataset())
        _ = al._labelled
        assert len(w) == 1
        assert w[0].category is DeprecationWarning


def test_labelled_map():
    ds = ActiveLearningDataset(MyDataset())
    assert ds.current_al_step == 0
    ds.label_randomly(10)
    assert ds.current_al_step == 1
    ds.label_randomly(10)
    assert ds.labelled_map.max() == 2 and np.equal(ds.labelled, ds.labelled_map > 0).all()

    st = ds.state_dict()
    ds2 = ActiveLearningDataset(MyDataset(), labelled=st["labelled"])
    assert ds2.current_al_step == ds.current_al_step


def test_last_active_step():
    ds = ActiveLearningDataset(MyDataset(), last_active_steps=1)
    assert len(ds) == 0
    ds.label_randomly(10)
    assert len(ds) == 10
    ds.label_randomly(10)
    # We only iterate over the items labelled at step 2.
    assert len(ds) == 10
    assert all(ds.labelled_map[x] == 2 for x, _ in ds)


if __name__ == '__main__':
    pytest.main()
