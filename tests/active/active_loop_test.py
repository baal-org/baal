import os
import pickle
import warnings

import numpy as np
import pytest
from torch.utils.data import Dataset

from baal.active import ActiveLearningDataset, heuristics
from baal.active.active_loop import ActiveLearningLoop

pjoin = os.path.join


class MyDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        pass

    def __len__(self):
        return 100

    def __getitem__(self, item):
        if self.transform:
            item = self.transform(item)
        return item


def get_probs(pool, dummy_param):
    acc = []
    for x in pool:
        b = np.zeros([1, 3, 10])
        b[0, x % 3, :] = 1
        acc.append(b)
    if acc:
        return np.vstack(acc)
    return None


def get_probs_iter(pool, dummy_param=None):
    assert dummy_param is not None
    if len(pool) == 0:
        return None
    for x in pool:
        b = np.zeros([1, 3, 10])
        b[:, x % 3, :] = 1
        yield b


@pytest.mark.parametrize('heur', [heuristics.Random(),
                                  heuristics.BALD(),
                                  heuristics.Entropy(),
                                  heuristics.Variance(reduction='sum')])
def test_should_stop(heur):
    dataset = ActiveLearningDataset(MyDataset(), make_unlabelled=lambda x: -1)
    active_loop = ActiveLearningLoop(dataset,
                                     get_probs,
                                     heur,
                                     query_size=10,
                                     dummy_param=1)
    dataset.label_randomly(10)
    step = 0
    for _ in range(15):
        flg = active_loop.step()
        step += 1
        if not flg:
            break

    assert step == 10


@pytest.mark.parametrize('heur', [heuristics.Random(),
                                  heuristics.BALD(),
                                  heuristics.Entropy(),
                                  heuristics.Variance(reduction='sum')])
def test_should_stop_iter(heur):
    dataset = ActiveLearningDataset(MyDataset(), make_unlabelled=lambda x: -1)
    active_loop = ActiveLearningLoop(dataset,
                                     get_probs_iter,
                                     heur,
                                     query_size=10,
                                     dummy_param=1)
    dataset.label_randomly(10)
    step = 0
    for _ in range(15):
        flg = active_loop.step()
        step += 1
        if not flg:
            break

    assert step == 10


@pytest.mark.parametrize('max_sample,expected', [(-1, 10), (5, 5), (200, 10)])
def test_sad(max_sample, expected):
    dataset = ActiveLearningDataset(MyDataset(), make_unlabelled=lambda x: -1)
    active_loop = ActiveLearningLoop(dataset,
                                     get_probs_iter,
                                     heuristics.Random(),
                                     max_sample=max_sample,
                                     query_size=10,
                                     dummy_param=1)
    dataset.label_randomly(10)
    active_loop.step()
    assert len(dataset) == 10 + expected


def test_file_saving(tmpdir):
    tmpdir = str(tmpdir)
    heur = heuristics.BALD()
    ds = MyDataset()
    dataset = ActiveLearningDataset(ds, make_unlabelled=lambda x: -1)
    active_loop = ActiveLearningLoop(dataset,
                                     get_probs_iter,
                                     heur,
                                     uncertainty_folder=tmpdir,
                                     query_size=10,
                                     dummy_param=1)
    dataset.label_randomly(10)
    _ = active_loop.step()
    assert len(os.listdir(tmpdir)) == 1
    file = pjoin(tmpdir, os.listdir(tmpdir)[0])
    assert "pool=90" in file and "labelled=10" in file
    data = pickle.load(open(file, 'rb'))
    assert len(data['uncertainty']) == 90
    # The diff between the current state and the step before is the newly labelled item.
    assert (data['dataset']['labelled'] != dataset.labelled).sum() == 10


def test_deprecation():
    heur = heuristics.BALD()
    ds = MyDataset()
    dataset = ActiveLearningDataset(ds, make_unlabelled=lambda x: -1)
    with warnings.catch_warnings(record=True) as w:
        active_loop = ActiveLearningLoop(dataset,
                                         get_probs_iter,
                                         heur,
                                         ndata_to_label=10,
                                         dummy_param=1)
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "ndata_to_label" in str(w[-1].message)

if __name__ == '__main__':
    pytest.main()
