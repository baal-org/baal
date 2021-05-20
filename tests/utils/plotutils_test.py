import sys

import numpy as np
import pytest
from baal.utils.plot_utils import make_animation_from_data


@pytest.mark.skipif(sys.platform == "darwin", reason="Does not work on Mac.")
def test_make_animation_from_data():
    x = np.random.rand(4, 2)
    y = np.random.rand(4)
    labelled_at = np.random.randint(0, 4, size=[x.shape[0]])
    classes = ['pos', 'neg']

    result = make_animation_from_data(x, y, labelled_at, classes)
    assert isinstance(result, list)
    assert result[0].shape[2] == 3


if __name__ == '__main__':
    pytest.main()
