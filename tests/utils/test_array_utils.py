import pytest
import torch

from baal.utils import array_utils
from baal.utils.iterutils import map_on_tensor


@pytest.fixture()
def a_tensor():
    return torch.randn([10, 3, 32, 32])


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


if __name__ == '__main__':
    pytest.main()
