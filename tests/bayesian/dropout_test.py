import warnings

import pytest
import torch

import baal.bayesian.dropout


@pytest.fixture
def a_model_with_dropout():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(), ),
        torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(5, 2),
        ))


def test_1d_eval_remains_stochastic():
    dummy_input = torch.randn(8, 10)
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        baal.bayesian.dropout.Dropout(p=0.5),
        torch.nn.Linear(5, 2),
    )
    test_module.eval()
    # NOTE: This is quite a stochastic test...
    torch.manual_seed(2019)
    with torch.no_grad():
        assert not all(
            (test_module(dummy_input) == test_module(dummy_input)).all()
            for _ in range(10)
        )


def test_2d_eval_remains_stochastic():
    dummy_input = torch.randn(8, 1, 5, 5)
    test_module = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 1),
        torch.nn.ReLU(),
        baal.bayesian.dropout.Dropout2d(p=0.5),
        torch.nn.Conv2d(1, 1, 1),
    )
    test_module.eval()
    # NOTE: This is quite a stochastic test...
    torch.manual_seed(2019)
    with torch.no_grad():
        assert not all(
            (test_module(dummy_input) == test_module(dummy_input)).all()
            for _ in range(10)
        )


@pytest.mark.parametrize("inplace", (True, False))
def test_patch_module_replaces_all_dropout_layers(inplace, a_model_with_dropout):
    mc_test_module = baal.bayesian.dropout.patch_module(a_model_with_dropout, inplace=inplace)

    # objects should be the same if inplace is True and not otherwise:
    assert (mc_test_module is a_model_with_dropout) == inplace
    assert not any(
        isinstance(module, torch.nn.Dropout) for module in mc_test_module.modules()
    )
    assert any(
        isinstance(module, baal.bayesian.dropout.Dropout)
        for module in mc_test_module.modules()
    )


@pytest.mark.parametrize("inplace", (True, False))
def test_patch_module_raise_warnings(inplace):
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    with warnings.catch_warnings(record=True) as w:
        mc_test_module = baal.bayesian.dropout.patch_module(test_module, inplace=inplace)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "No layer was modified by patch_module" in str(w[-1].message)


def test_module_class_replaces_dropout_layers(a_model_with_dropout, is_deterministic):
    dummy_input = torch.randn(8, 10)
    test_mc_module = baal.bayesian.dropout.MCDropoutModule(a_model_with_dropout)

    assert not any(
        isinstance(module, torch.nn.Dropout) for module in a_model_with_dropout.modules()
    )
    assert any(
        isinstance(module, baal.bayesian.dropout.Dropout)
        for module in a_model_with_dropout.modules()
    )
    torch.manual_seed(2019)
    with torch.no_grad():
        assert not all(
            (test_mc_module(dummy_input) == test_mc_module(dummy_input)).all()
            for _ in range(10)
        )


    # Check that unpatch works
    module = test_mc_module.unpatch().eval()
    assert not any(isinstance(mod, baal.bayesian.dropout.Dropout) for mod in module.modules())
    assert is_deterministic(module, (8, 10))



if __name__ == '__main__':
    pytest.main()
