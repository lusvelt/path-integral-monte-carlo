import numpy as np

def get_linspace_idxs_within(x_arr: np.ndarray, x_min: float) -> np.ndarray:
    """
    Computes the indices of the elements of a linspace within a larger linspace with equal spacing.

    Args:
        x_arr (numpy.ndarray): linspace of which we want to compute indices within the larger linspace.
        x_min (float): Start of larger linspace.

    Returns
        numpy.ndarray[int]: numpy.arange of indices of the elements of `x_arr` within `numpy.linspace(x_min, x_max, ...)` having elements spaced as in `x_arr`.
    """
    diff = np.diff(x_arr)
    assert np.all(np.where(diff == diff[0]))
    dx = np.diff[0]
    idx_start = int(np.round((x_arr[0] - x_min) / dx))
    idx_end = int(np.round((x_arr[-1] - x_min) / dx))
    return np.arange(idx_start, idx_end + 1)


def get_extended_linspace_size(x_arr: np.ndarray, extent: float) -> int:
    """
    Computes the size of an extended linspace having spacing equal to a given one.

    Args:
        x_arr (numpy.ndarray): linspace we want to extend.
        extent (float): Span of extended linspace.

    Returns
        int: Size of extended linspace
    """
    x_range = x_arr[-1] - x_arr[0]
    return int(np.round((x_arr.shape[0] - 1) / x_range * extent)) + 1