"""DISCLAIMER

This work has been done by Jan Freyberg here: https://github.com/janfreyberg/torch-hypothesis
But this repo is no longer maintained so we copied it here.
All credits goes to Jan Freyberg.
"""

from typing import Any, Sequence
import torch
import numpy as np

from hypothesis import strategies as st
from hypothesis.extra import numpy as npst


def torch_dtype_to_numpy_dtype(dtype):
    return np.dtype(str(dtype).replace("torch.", ""))


def is_numeric(val: Any) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


def from_range_value_or_choice(parameter, draw):
    # if strategy, just draw:
    if isinstance(parameter, st.SearchStrategy):
        return draw(parameter)
    # if len 2 sequence of numbers, draw from strat:
    if (
        isinstance(parameter, Sequence)
        and len(parameter) == 2
        and all(is_numeric(param) or param is None for param in parameter)
    ):
        if all(isinstance(param, int) or param is None for param in parameter):
            return draw(
                st.integers(min_value=parameter[0], max_value=parameter[1])
            )
        elif all(
            isinstance(param, float) or param is None for param in parameter
        ):
            return draw(
                st.floats(min_value=parameter[0], max_value=parameter[1])
            )
    # if some other sequence, return one element from it:
    elif isinstance(parameter, Sequence):
        return draw(st.sampled_from(parameter))
    else:
        return parameter


@st.composite
def torch_tensor(draw, shape, values, dtype):
    shape = (from_range_value_or_choice(size, draw) for size in shape)
    dtype = from_range_value_or_choice(dtype, draw)
    np_array = draw(
        npst.arrays(torch_dtype_to_numpy_dtype(dtype), shape, elements=values)
    )
    return torch.as_tensor(np_array)


@st.composite
def linear_layer_input(
    draw,
    batch_size=(1, 256),
    dimensionality=(1, 2000),
    values=st.floats(allow_nan=False, allow_infinity=False, width=32),
    dtype=torch.float,
):
    """Produce linear layer input.

    Parameters
    ----------
    batch_size : tuple, optional
        The batch size. Can be an integer, a tuple defining a range of ints
        (e.g. (1, 32)), or a hypothesis strategy.
    dimensionality : tuple, optional
        [description], by default (1, 2000)
    values : [type], optional
    dtype : [type], optional
        [description], by default torch.float

    Returns
    -------
    torch.Tensor
        batch_size x dimensionality
    """
    return draw(
        torch_tensor((batch_size, dimensionality), values, torch.float)
    )


@st.composite
def class_logits(draw, batch_size=(1, 256), n_classes=(1, 2000)):
    """Return logits - the standard output of a standard classification net.

    Parameters
    ----------
    batch_size : tuple, optional
        [description], by default (1, 256)
    n_classes : tuple, optional
        [description], by default (1, 2000)

    Returns
    -------
    torch.Tensor
        A float-tensor of size batch_size x n_classes.
    """
    logits = draw(
        linear_layer_input(batch_size=batch_size, dimensionality=n_classes)
    )
    logits = logits
    return logits


@st.composite
def class_labels(draw, batch_size=(1, 256), n_classes=(1, 2000)):
    """[summary]

    Parameters
    ----------
    batch_size : tuple, optional
        [description], by default (1, 256)
    n_classes : tuple, optional
        [description], by default (1, 2000)
    """

    shape = (from_range_value_or_choice(batch_size, draw),)
    n_classes = from_range_value_or_choice(n_classes, draw)
    labels = npst.arrays(
        dtype=int, shape=shape, elements=st.integers(min_value=0, max_value=n_classes - 1)
    )
    return torch.as_tensor(draw(labels)).to(torch.long)


@st.composite
def classification_logits_and_labels(
    draw, batch_size=(1, 256), n_classes=(1, 2000)
):
    batch_size = from_range_value_or_choice(batch_size, draw)
    n_classes = from_range_value_or_choice(n_classes, draw)
    return (
        draw(class_logits(batch_size=batch_size, n_classes=n_classes)),
        draw(class_labels(batch_size=batch_size, n_classes=n_classes)),
    )
