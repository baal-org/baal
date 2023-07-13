import pytest
import torch
from torch.nn import Sequential, Linear

from baal.bayesian.caching_utils import MCCachingModule


class LinearMocked(Linear):
    call_count = 0

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

    def __call__(self, x):
        LinearMocked.call_count += 1
        return super().__call__(x)


@pytest.fixture()
def my_model():
    return Sequential(
        LinearMocked(10, 10),
        LinearMocked(10, 10),
        Sequential(
            LinearMocked(10, 10),
            LinearMocked(10, 10),
        )
    ).eval()


def test_caching(my_model):
    x = torch.rand(10)

    # No Caching
    my_model(x)
    assert LinearMocked.call_count == 4
    my_model(x)
    assert LinearMocked.call_count == 8

    with MCCachingModule(my_model) as model:
        model(x)
        assert LinearMocked.call_count == 12
        model(x)
        assert LinearMocked.call_count == 12

    # No Caching
    my_model(x)
    assert LinearMocked.call_count == 16
    my_model(x)
    assert LinearMocked.call_count == 20


