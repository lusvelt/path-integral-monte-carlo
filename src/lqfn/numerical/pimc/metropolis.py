"""
This module provides functions to compute path integrals using Metropolis Monte Carlo method (explained in section 2.3 of Lepage's paper "Lattice QCD for Novices")
"""

from typing import Callable
import numpy as np


def _update_path(path, S, eps):
    N = path.shape[0]
    for j in range(N):
        old_x = path[j]  # save original value
        old_Sj = S(j, path)
        path[j] = path[j] + np.random.uniform(-eps, eps)  # update path[j]
        dS = S(j, path) - old_Sj  # change in action
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            path[j] = old_x  # restore old value


def compute_path_integral_average(
    functional: Callable,
    S_per_component: Callable,
    N: int,
    N_cf: int,
    N_cor: int,
    eps: float,
    thermalization_its: int = 5,
):
    """
    Computes the path integral average

    $$\\langle\\langle \\Gamma[x] \\rangle\\rangle = \\frac{\\int Dx \\Gamma[x] e^{-S[x]}}{\\int Dx e^{-S[x]}}$$

    using the Metropolis Monte Carlo method explained in section 2.2 of "Lattice QCD for Novices" of P. Lepage

    Args:
        functional(Callable[numpy.ndarray[float, N], float]): A functional taking a path as input (as a numpy array of length N) and returning a number.
        S_per_component(Callable[int, numpy.ndarray[float, N]]): A functional taking as input an integer $j$ and a path, and returning the contribution of the $j$-th component of each path point to the action.
        N (int): Number of path points.
        N_cf (int): Total number of samples contributing to be saved during the process for computing the path integral average.
        N_cor (int): Number of path updates before picking each sample.
        eps (float): $\\epsilon$ parameter for the update of the path.
        thermalization_its (int): Number of samples to be discarded at the beginning to let the procedure thermalize. Default is 10.

    Returns:
        numpy.ndarray[float, N]: Path integral average of the functional.
        numpy.ndarray[float, N]: Standard deviation error associated to each instant of time in the path integral average.
    """
    functional_samples = np.zeros((N_cf, N))
    path = np.zeros(N)
    for j in range(N):
        path[j] = 0
    for j in range(thermalization_its * N_cor):  # thermalization
        _update_path(path, S_per_component, eps)
    for rows in range(N_cf):
        for j in range(N_cor):  # discard N_cor values
            _update_path(path, S_per_component, eps)
        for n in range(N):
            functional_samples[rows][n] = functional(
                path, n
            )  # for every time instant we have N_cf values of G
    avg = functional_samples.mean(axis=0)
    std = functional_samples.std(axis=0) / np.sqrt(N_cf)
    return avg, std
