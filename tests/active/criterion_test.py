from baal.active.stopping_criteria import (
    LabellingBudgetStoppingCriterion,
    EarlyStoppingCriterion,
    LowAverageUncertaintyStoppingCriterion,
)
from baal.active.dataset import ActiveNumpyArray
import numpy as np


def test_labelling_budget():
    ds = ActiveNumpyArray((np.random.randn(100, 3), np.random.randint(0, 3, 100)))
    ds.label_randomly(10)
    criterion = LabellingBudgetStoppingCriterion(ds, labelling_budget=50)
    assert not criterion.should_stop({}, [])

    ds.label_randomly(10)
    assert not criterion.should_stop({}, [])

    ds.label_randomly(40)
    assert criterion.should_stop({}, [])


def test_early_stopping():
    ds = ActiveNumpyArray((np.random.randn(100, 3), np.random.randint(0, 3, 100)))
    criterion = EarlyStoppingCriterion(ds, "test_loss", patience=5)

    for i in range(10):
        assert not criterion.should_stop(
            metrics={"test_loss": 1 / (i + 1)}, uncertainty=[]
        )

    for _ in range(4):
        assert not criterion.should_stop(metrics={"test_loss": 0.1}, uncertainty=[])
    assert criterion.should_stop(metrics={"test_loss": 0.1}, uncertainty=[])

    # test less than patience stability
    criterion = EarlyStoppingCriterion(ds, "test_loss", patience=5)
    for _ in range(4):
        assert not criterion.should_stop(metrics={"test_loss": 0.1}, uncertainty=[])
    assert criterion.should_stop(metrics={"test_loss": 0.1}, uncertainty=[])


def test_low_average():
    ds = ActiveNumpyArray((np.random.randn(100, 3), np.random.randint(0, 3, 100)))
    criterion = LowAverageUncertaintyStoppingCriterion(
        active_dataset=ds, avg_uncertainty_thresh=0.1
    )
    assert not criterion.should_stop(
        metrics={}, uncertainty=np.random.normal(0.5, scale=0.8, size=(100,))
    )
    assert criterion.should_stop(
        metrics={}, uncertainty=np.random.normal(0.05, scale=0.01, size=(100,))
    )
