"""
This module contains utilities for plotting graphs.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_delta_E(exact: float, data: np.ndarray, err: np.ndarray, a: float, title: str, n: int = None):
    """
    Creates the plot of $\\Delta E_n$ with error bars.

    Args:
        exact (float): The expected exact value for $\\Delta E$
        data (np.ndarray): A numpy array containing the computed values of $\\Delta E_n$ for each $n$
        err (np.ndarray): A numpy array containing the errors associated to each $\\Delta E_n$ for each $n$
        a (float): The temporal lattice spacing (affects the ticks on the horizontal axis)
        title (str): The title of the plot
        n (int): The number $n$ of points to be shown, from $\\Delta E_0$ to $\\Delta E_{n-1}$

    Returns:
        matplotlib.Figure: The figure `fig` containing the plot, to be shown by calling `fig.show()`
    """
    assert data.shape == err.shape
    assert len(data.shape) == 1
    N = data.shape[0]
    assert n is None or (0 < n <= N)
    if n is None:
        n = N
    t = a * np.arange(n)
    fig, ax = plt.subplots()
    ax.plot(t, exact * np.ones(n), color="blue", label="Expectation")
    ax.errorbar(t, data[:n], yerr=err[:n], fmt=".", color="black", label="Numerical")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\Delta E(t)$")
    ax.legend()
    return fig
