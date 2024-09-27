import pytest
from lqfn.utils import *
import numpy as np


def test_get_linspace_idxs_within():
    small_linspace = np.linspace(0, 1, 6)
    # [0, 0.2, 0.4, 0.6, 0.8, 1]
    # Within a linspace starting from -2, the indices should be:
    correct_idxs = np.arange(10, 15 + 1)
    idxs = get_linspace_idxs_within(small_linspace, -2)
    assert np.array_equal(idxs, correct_idxs)

    # If the array is not evenly spaced, should raise AssertionError
    log_space = np.logspace(1, 10, 5)
    with pytest.raises(AssertionError):
        get_linspace_idxs_within(log_space, 0.5)


def test_get_extended_linspace_size():
    linspace = np.linspace(0, 1, 6)
    extent = 3
    correct_N = 16
    computed_N = get_extended_linspace_size(linspace, extent)
    assert correct_N == computed_N

    # If the array is not evenly spaced, should raise AssertionError
    log_space = np.logspace(1, 10, 5)
    with pytest.raises(AssertionError):
        get_extended_linspace_size(log_space, 20)
