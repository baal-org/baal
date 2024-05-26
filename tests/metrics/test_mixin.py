import numpy as np

from baal.modelwrapper import ModelWrapper, TrainingArgs
from baal.utils.metrics import Accuracy


def test_active_step():
    wrapper = ModelWrapper(None, TrainingArgs())
    precisions = np.linspace(0, 1, 10, endpoint=False)
    recalls = np.linspace(0.5, 1, 10, endpoint=False)
    dataset_size = list(range(100, 1100, 100))
    for ds_size, precision, recall in zip(dataset_size, precisions, recalls):
        wrapper.active_step(ds_size, {
            'Precision': precision,
            'Recall': recall
        })
    assert len(wrapper.active_learning_metrics) == 10
    assert wrapper.active_learning_metrics[100] == {
        'Precision': 0.0,
        'Recall': 0.5
    }

    wrapper = ModelWrapper(None, TrainingArgs())
    wrapper.set_dataset_size(1000)
    wrapper.active_step(dataset_size=None, metrics={
        'Precision': 0.1,
        'Recall': 0.2
    })
    assert wrapper._active_dataset_size == 1000
    assert wrapper.active_learning_metrics == {
        1000: {
            'Precision': 0.1,
            'Recall': 0.2
        }
    }


def test_get_metrics():
    wrapper = ModelWrapper(None, TrainingArgs())
    wrapper.add_metric('accuracy', Accuracy)

    assert len(wrapper.get_metrics()) == 4
    assert len(wrapper.get_metrics('test')) == 2
    assert all('test' in ki for ki in wrapper.get_metrics('test'))
    assert len(wrapper.get_metrics('train')) == 2
    assert all('train' in ki for ki in wrapper.get_metrics('train'))

    wrapper.set_dataset_size(1000)
    assert len(wrapper.get_metrics()) == 5
    assert len(wrapper.get_metrics('test')) == 3
    assert sum('test' in ki for ki in wrapper.get_metrics('test')) == 2
    assert len(wrapper.get_metrics('train')) == 3
    assert sum('train' in ki for ki in wrapper.get_metrics('train')) == 2
