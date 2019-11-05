import pytest
import numpy as np
import torch

from baal.bayesian.dropconnect import MCDropoutConnectModule, DropConWrapper


def test_dropconwrapper():
    dummy_input = torch.randn(8, 10)
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    mc_test_module = MCDropoutConnectModule(test_module, layers=['Linear'], weight_dropout=0.2)
    mc_test_module = DropConWrapper(mc_test_module, criterion=None)
    mc_test_module.eval()

    out = mc_test_module.predict_on_batch(dummy_input, iterations=10, cuda=False)
    assert out.size() == torch.Size([8, 2, 10])

    out1 = np.array(mc_test_module.predict_on_batch(dummy_input, iterations=1, cuda=False))
    out2 = np.array(mc_test_module.predict_on_batch(dummy_input, iterations=1, cuda=False))
    assert not np.allclose(out1, out2)

    out = np.array(mc_test_module.predict_on_batch(dummy_input, iterations=2, cuda=False))
    assert not np.allclose(out[..., 0], out[..., 1])


if __name__ == '__main__':
    pytest.main()
