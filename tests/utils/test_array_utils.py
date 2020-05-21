import pytest
import torch
from scipy.special import softmax, expit

from baal.utils import array_utils
from baal.utils.array_utils import to_prob
from baal.utils.iterutils import map_on_tensor
import numpy as np

@pytest.fixture()
def a_tensor():
    return torch.randn([10, 3, 32, 32])

@pytest.fixture()
def an_array():
    return np.random.randn(10, 3, 32, 32)

@pytest.fixture()
def a_binary_array():
    return np.random.randn(10, 1, 32, 32)


def test_stack_in_memory_single(a_tensor):
    iterations = 10
    out = array_utils.stack_in_memory(a_tensor, iterations=iterations)
    assert out.shape == (10 * iterations, 3, 32, 32)


def test_stack_in_memory_multi(a_tensor):
    iterations = 10
    t = [a_tensor, a_tensor]
    out = map_on_tensor(lambda ti: array_utils.stack_in_memory(ti, iterations=iterations), t)
    assert out[0].shape == (10 * iterations, 3, 32, 32)
    assert out[1].shape == (10 * iterations, 3, 32, 32)

def test_to_prob(an_array, a_binary_array):
    out = to_prob(an_array)
    assert not np.allclose(out, an_array)

    out = to_prob(a_binary_array)
    assert not np.allclose(out, a_binary_array)

    a_array_scaled = softmax(an_array, 1)
    a_binary_array_scaled = expit(a_binary_array)

    out = to_prob(a_array_scaled)
    assert np.allclose(out, a_array_scaled)

    out = to_prob(a_binary_array_scaled)
    assert np.allclose(out, a_binary_array_scaled)





if __name__ == '__main__':
    pytest.main()
