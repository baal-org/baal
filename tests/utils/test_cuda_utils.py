import os
import pytest
import torch
from collections import namedtuple, OrderedDict

from baal.utils.cuda_utils import to_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_to_cuda():
    t = torch.randn(3, 5)
    assert to_cuda(t).device.type == "cuda"
    m = torch.nn.Linear(3, 1)
    assert next(to_cuda(m).parameters()).device.type == "cuda"

    t_list = [t, t.clone()]
    assert type(t_list) is type(to_cuda(t_list))
    assert all(t_.device.type == "cuda" for t_ in to_cuda(t_list))

    t_dict = {"t": t, "t2": t.clone()}
    assert type(t_dict) is type(to_cuda(t_dict))
    assert all(t_.device.type == "cuda" for t_ in to_cuda(t_dict).values())

    MyTuple = namedtuple("MyTuple", ["t", "t2"])
    t_named_tuple = MyTuple(t, t.clone())
    assert type(t_named_tuple) is type(to_cuda(t_named_tuple))
    assert all(t_.device.type == "cuda" for t_ in to_cuda(t_named_tuple))

    t_tuple = (t, t.clone())
    assert type(t_tuple) is type(to_cuda(t_tuple))
    assert all(t_.device.type == "cuda" for t_ in to_cuda(t_tuple))

    # test a type that's not explicitly defined:
    t_ordered_dict = OrderedDict([("t", t), ("t2", t.clone())])
    assert type(t_ordered_dict) is type(to_cuda(t_ordered_dict))
    assert all(t_.device.type == "cuda" for t_ in to_cuda(t_ordered_dict).values())

    # test strings
    t_ordered_dict = OrderedDict([("t", "string2"), ("t2", "string1")])
    assert type(t_ordered_dict) is type(to_cuda(t_ordered_dict))
    assert to_cuda(t_ordered_dict)['t'] == "string2"
