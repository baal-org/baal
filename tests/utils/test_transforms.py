import pytest
import numpy as np
import torch
from PIL import Image
from hypothesis.extra import numpy as np_strategies
from hypothesis import given

from baal.utils.transforms import PILToLongTensor


@given(img=np_strategies.arrays(
    np.uint8,
    (100, 100, 3)
))
def test_pil_to_long_tensor(img):

    transformer = PILToLongTensor(
        classes=[np.array([100, 100, 100]), np.array([101, 102, 104])]
    )
    # test with numpy:
    long_img = transformer(img)
    assert isinstance(long_img, torch.Tensor)
    # test with PIL
    img = Image.fromarray(img)
    long_img_2 = transformer(img)
    assert isinstance(long_img, torch.Tensor)
    assert (long_img == long_img_2).all()
