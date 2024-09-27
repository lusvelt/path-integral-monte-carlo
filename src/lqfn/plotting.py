"""
This module contains utilities for plotting graphs.
"""

import numpy as np
import matplotlib.pyplot as plt


class Plot:
    """
    This class represents a matplotlib plot. One can manage the objects that appear in the plot by invoking the functions that are present in this class.
    """

    def __init__(self, xlabel, ylabel, title):
        """
        Args:
            title (str): The title of the plot.
            xlabel (str): The label for the x axis.
            ylabel (str): The label for the y axis.
        """
        fig, ax = plt.subplots()
        self.fig = fig
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.ax = ax

    def add_constant_plot(self, a, b, value, color, label):
        """
        Adds a plot of a constant value (horizontal line) to the plot.

        Args:
            a (float): The left bound of the plot.
            b (float): The right bound of the plot.
            value (float): The value of the constant.
            color (string): The color of the plot.
        """
        self.ax.plot(np.array([a, b]), value * np.ones(2), color=color, label=label)

    def add_errorbar_points(self, x, y, yerr, color, label):
        """
        Adds points with error bars to the plot.

        Args:
            x (np.ndarray): The x coordinates of the points.
            y (np.ndarray): The y coordinates of the points.
            yerr (np.ndarray): The errors associated to the points, in the y axis.
            color (str): The color of the points.
            label (str): The label of the dataset.
        """
        self.ax.errorbar(x, y, yerr=yerr, fmt=".", color=color, label=label)

    def show(self):
        """
        Adds the legend and shows the plot.
        """
        self.ax.legend()
        self.fig.show()


def plot_delta_E(
    data: np.ndarray,
    err: np.ndarray,
    exact: float,
    a: float,
    title: str,
    n: int = None,
    fit: float = None,
):
    """
    Creates the plot of $\\Delta E_n$ with error bars.

    Args:
        data (np.ndarray): A numpy array containing the computed values of $\\Delta E_n$ for each $n$
        err (np.ndarray): A numpy array containing the errors associated to each $\\Delta E_n$ for each $n$
        exact (float): The expected exact value for $\\Delta E$
        a (float): The temporal lattice spacing (affects the ticks on the horizontal axis)
        title (str): The title of the plot
        n (int, optional): The number $n$ of points to be shown, from $\\Delta E_0$ to $\\Delta E_{n-1}$. Default is maximum.
        fit (float, optional): The value of $\\Delta E$ estimated by path integrals

    Returns:
        Plot: An oblect of the class `lqfn.plotting.Plot` representing the plot.
    """
    assert data.shape == err.shape
    assert len(data.shape) == 1
    N = data.shape[0]
    assert n is None or (0 < n <= N)
    if n is None:
        n = N
    t = a * np.arange(n)
    plot = Plot(title=title, xlabel="t", ylabel=r"$\Delta E$")
    plot.add_constant_plot(0, a * (n - 1), exact, color="blue", label="exact")
    if fit is not None:
        plot.add_constant_plot(0, a * (n - 1), fit, color="green", label="path integral")
    plot.add_errorbar_points(t, data[:n], yerr=err[:n], color="black", label="points")
    return plot
