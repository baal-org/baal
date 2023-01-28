import pytest
from torch import nn

from baal.bayesian.common import replace_layers_in_module


@pytest.fixture
def a_model_deep():
    return nn.Sequential(
        nn.Linear(32, 32),
        nn.Sequential(
            nn.Linear(32, 3),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.ReLU(),
            nn.Linear(3, 3)
        ))


@pytest.fixture
def a_model():
    return nn.Sequential(
        nn.Linear(32, 3),
        nn.ReLU(),
        nn.Linear(10, 3),
        nn.ReLU(),
        nn.Linear(3, 3)
    )


def test_replace_layers_in_module_swap_all_relu(a_model):
    mapping = lambda mod: None if not isinstance(mod, nn.ReLU) else nn.Identity()
    changed = replace_layers_in_module(a_model, mapping)
    assert changed
    assert not any(isinstance(m, nn.ReLU) for m in a_model.modules())
    assert any(isinstance(m, nn.Identity) for m in a_model.modules())


def test_replace_layers_in_module_swap_all_relu_deep(a_model_deep):
    mapping = lambda mod: None if not isinstance(mod, nn.ReLU) else nn.Identity()
    changed = replace_layers_in_module(a_model_deep, mapping)
    assert changed
    assert not any(isinstance(m, nn.ReLU) for m in a_model_deep.modules())
    assert any(isinstance(m, nn.Identity) for m in a_model_deep.modules())


def test_replace_layers_in_module_swap_no_relu_deep(a_model_deep):
    mapping = lambda mod: None if not isinstance(mod, nn.ReLU6) else nn.Identity()
    changed = replace_layers_in_module(a_model_deep, mapping)
    assert not changed
    assert any(isinstance(m, nn.ReLU) for m in a_model_deep.modules())
    assert not any(isinstance(m, nn.Identity) for m in a_model_deep.modules())

def test_replace_layers_in_module_swap_no_relu_deep(a_model):
    mapping = lambda mod: None if not isinstance(mod, nn.ReLU6) else nn.Identity()
    changed = replace_layers_in_module(a_model, mapping)
    assert not changed
    assert any(isinstance(m, nn.ReLU) for m in a_model.modules())
    assert not any(isinstance(m, nn.Identity) for m in a_model.modules())


@pytest.mark.parametrize('state', [True, False])
def test_training_state(a_model, state):
    # Check that the state of the Module is the same as previously.
    a_model = a_model.train(mode=state)
    mapping = lambda mod: None if not isinstance(mod, nn.ReLU) else nn.Identity()
    _ = replace_layers_in_module(a_model, mapping)
    assert all(m.training is state for m in a_model.modules())

    unmap = lambda mod: None if not isinstance(mod, nn.Identity) else nn.ReLU()
    _ = replace_layers_in_module(a_model, unmap)
    assert all(m.training is state for m in a_model.modules())



if __name__ == '__main__':
    pytest.main()
