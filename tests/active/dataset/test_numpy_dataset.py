from sklearn.datasets import load_iris

from baal.active.dataset import ActiveNumpyArray
import numpy as np

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