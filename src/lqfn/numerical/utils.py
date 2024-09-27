"""
This is a utility module that provides some useful functions.
"""

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
    assert np.all(np.isclose(diff, diff[0]))
    dx = diff[0]
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
    diff = np.diff(x_arr)
    assert np.all(np.isclose(diff, diff[0]))
    x_range = x_arr[-1] - x_arr[0]
    return int(np.round((x_arr.shape[0] - 1) / x_range * extent)) + 1


def get_weighted_avg_and_err(data: np.ndarray, errs: np.ndarray, N_points: int):
    """
    Computes the weighted average of a dataset, and the corresponding standard deviation.

    Args:
        data (np.ndarray): Numpy array containing the data points
        err (np.ndarray): Numpy array containing the errors associated to the data points
        N_points (int): Number of points to fetch from the array to compute the weighted average

    Returns:
        float: The weighted average
        float: The error associated to the weighted average
    """
    assert data.shape == errs.shape
    avg = np.average(data[:N_points], weights=1 / errs[:N_points] ** 2)
    err = np.sqrt(np.average((data[:N_points] - avg) ** 2))
    return avg, err
