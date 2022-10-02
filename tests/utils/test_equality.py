from collections import namedtuple

import numpy as np
import torch

from baal.utils.equality import deep_check

Point = namedtuple('Point', 'x,y')


def test_deep_check():
    arr1, arr2 = np.random.rand(10), np.random.rand(10)
    tensor1, tensor2 = torch.rand([10]), torch.rand([10])
    s1, s2 = "string1", "string2"
    p1, p2 = Point(x=1, y=2), Point(x=2, y=1)

    assert not deep_check(arr1, arr2)
    assert not deep_check(tensor1, tensor2)
    assert not deep_check(s1, s2)
    assert not deep_check(p1, p2)
    assert not deep_check([arr1, tensor1], [arr2, tensor2])
    assert not deep_check([arr1, tensor1], (arr1, tensor1))
    assert not deep_check([arr1, tensor1], [tensor1, arr1])
    assert not deep_check({'x': arr1, 'y': tensor1}, {'x': arr2, 'y': tensor2})
    assert not deep_check({'x': arr1, 'y': tensor1}, {'x': tensor1, 'y': arr1})

    assert deep_check(arr1, arr1)
    assert deep_check(tensor1, tensor1)
    assert deep_check(s1, s1)
    assert deep_check(p1, p1)
    assert deep_check([arr1, tensor1], [arr1, tensor1])
    assert deep_check((arr1, tensor1), (arr1, tensor1))
    assert deep_check({'x': arr1, 'y': tensor1}, {'x': arr1, 'y': tensor1})
