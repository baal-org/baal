import numpy as np
import pytest
import torch

from baal.utils.iterutils import map_on_tensor


def test_map_on_tensor():
    x = [torch.zeros([10]), torch.zeros([10])]
    assert np.allclose(map_on_tensor(lambda xi: (xi + 1).numpy(), x),
                       [np.ones([10]), np.ones([10])])
    x = torch.zeros([10])
    assert np.allclose(map_on_tensor(lambda xi: xi + 1, x).numpy(), np.ones([10]))


if __name__ == '__main__':
    pytest.main()
