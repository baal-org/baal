import warnings
from copy import deepcopy

import pytest
import torch

import baal.bayesian.consistent_dropout


def test_1d_eval_is_not_stochastic():
    dummy_input = torch.randn(8, 10)
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        baal.bayesian.consistent_dropout.ConsistentDropout(p=0.5),
        torch.nn.Linear(5, 2),
    )
    test_module.eval()
    # NOTE: This is quite a stochastic test...
    torch.manual_seed(2019)
    with torch.no_grad():
        assert all(
            (test_module(dummy_input) == test_module(dummy_input)).all()
            for _ in range(10)
        )


def test_2d_eval_is_stochastic():
    dummy_input = torch.randn(8, 1, 5, 5)
    test_module = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 1),
        torch.nn.ReLU(),
        baal.bayesian.consistent_dropout.ConsistentDropout2d(p=0.5),
        torch.nn.Conv2d(1, 1, 1),
    )
    test_module.eval()
    # NOTE: This is quite a stochastic test...
    torch.manual_seed(2019)
    with torch.no_grad():
        assert all(
            (test_module(dummy_input) == test_module(dummy_input)).all()
            for _ in range(10)
        )


@pytest.mark.parametrize("inplace", (True, False))
def test_patch_module_replaces_all_dropout_layers(inplace):
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(5, 2),
    )

    mc_test_module = baal.bayesian.consistent_dropout.patch_module(test_module, inplace=inplace)

    # objects should be the same if inplace is True and not otherwise:
    assert (mc_test_module is test_module) == inplace
    assert not any(
        isinstance(module, torch.nn.Dropout) for module in mc_test_module.modules()
    )
    assert any(
        isinstance(module, baal.bayesian.consistent_dropout.ConsistentDropout)
        for module in mc_test_module.modules()
    )


def test_module_class_replaces_dropout_layers():
    dummy_input = torch.randn(8, 10)
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(5, 2),
    )
    test_mc_module = baal.bayesian.consistent_dropout.MCConsistentDropoutModule(test_module)

    assert not any(
        isinstance(module, torch.nn.Dropout) for module in test_module.modules()
    )
    assert any(
        isinstance(module, baal.bayesian.consistent_dropout.ConsistentDropout)
        for module in test_module.modules()
    )
    torch.manual_seed(2019)
    with torch.no_grad():
        assert not all(
            torch.eq(test_mc_module(dummy_input), test_mc_module(dummy_input)).all()
            for _ in range(10)
        )

    test_mc_module.eval()
    torch.manual_seed(2019)
    with torch.no_grad():
        assert all(
            torch.eq(test_mc_module(dummy_input), test_mc_module(dummy_input)).all()
            for _ in range(10)
        )

@pytest.mark.parametrize("inplace", (True, False))
def test_patch_module_raise_warnings(inplace):

    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    with warnings.catch_warnings(record=True) as w:
        mc_test_module = baal.bayesian.consistent_dropout.patch_module(test_module, inplace=inplace)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "No layer was modified by patch_module" in str(w[-1].message)

def test_intra_batch_is_stochastic():
    dummy_input = torch.randn(10)
    dummy_input = torch.stack([dummy_input, dummy_input])
    assert dummy_input.shape == (2, 10)
    test_module = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        baal.bayesian.consistent_dropout.ConsistentDropout(p=0.5),
        torch.nn.Linear(5, 2),
    )
    test_module.eval()
    # NOTE: This is quite a stochastic test...
    torch.manual_seed(2019)
    pred = test_module(dummy_input)
    assert (pred[0] - pred[1]).abs().sum() > 0

    assert torch.eq(pred, test_module(dummy_input)).all()


def test_masks_changes_1d():
    i = torch.randn([10, 223])
    l = baal.bayesian.consistent_dropout.ConsistentDropout()
    _mask_tests(i, l)


def test_masks_changes_2d():
    i = torch.randn([10, 3, 223, 223])
    l = baal.bayesian.consistent_dropout.ConsistentDropout2d()
    _mask_tests(i, l)


def _mask_tests(i, l):
    l.train()
    l.eval()
    assert l._mask is None
    o1 = l(i)
    mask = deepcopy(l._mask)

    l.train()
    l.eval()
    assert l._mask is None
    o2 = l(i)
    mask2 = deepcopy(l._mask)

    assert not torch.eq(mask, mask2).all()
    assert not torch.eq(o1, o2).all()
