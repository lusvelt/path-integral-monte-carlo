"""
This module provides functions to compute path integrals using Metropolis Monte Carlo method (explained in section 2.3 of Lepage's paper "Lattice QCD for Novices")
"""

from typing import Callable
import numpy as np


def _update_path(path, S_per_timeslice, eps):
    N = path.shape[0]
    for j in range(N):
        old_x = path[j]  # save original value
        old_Sj = S_per_timeslice(j, path)
        path[j] += np.random.uniform(-eps, eps)  # update path[j]
        dS = S_per_timeslice(j, path) - old_Sj  # change in action
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            path[j] = old_x  # restore old value


def _generate_functional_samples(
    functional: Callable,
    S_per_timeslice: Callable,
    N: int,
    N_cf: int,
    N_cor: int,
    eps: float,
    thermalization_its: int = 5,
):
    functional_samples = np.zeros((N_cf, N))
    path = np.zeros(N)
    for _ in range(thermalization_its * N_cor):  # thermalization
        _update_path(path, S_per_timeslice, eps)
    for rows in range(N_cf):
        for _ in range(N_cor):  # discard N_cor values
            _update_path(path, S_per_timeslice, eps)
        for n in range(N):  # for every time instant we have N_cf values of G
            functional_samples[rows][n] = functional(path, n)
    return functional_samples


def compute_path_integral_average(
    functional: Callable, S_per_timeslice: Callable, N: int, N_cf: int, N_cor: int, eps: float, thermalization_its: int = 5, bootstrap: bool = False
):
    """
    Computes the path integral average

    $$\\langle\\langle \\Gamma[x] \\rangle\\rangle = \\frac{\\int Dx \\Gamma[x] e^{-S[x]}}{\\int Dx e^{-S[x]}}$$

    using the Metropolis Monte Carlo method explained in section 2.2 of "Lattice QCD for Novices" of P. Lepage

    Args:
        functional(Callable[numpy.ndarray[float, N], float]): A functional taking a path (as a numpy array of length N) and a (discretized) time instant and returning a number.
        S_per_timeslice(Callable[int, numpy.ndarray[float, N]]): A functional taking as input an integer $j$ and a path, and returning the contribution of the $j$-th point of the path to the action.
        N (int): Number of path points.
        N_cf (int): Total number of samples contributing to be saved during the process for computing the path integral average.
        N_cor (int): Number of path updates before picking each sample.
        eps (float): $\\epsilon$ parameter for the update of the path.
        thermalization_its (int): Number of samples to be discarded at the beginning to let the procedure thermalize. Default is 5.
        bootstrap (bool): Whether to use bootstrap procedure to compute the average and the error.

    Returns:
        numpy.ndarray[float, N]: Path integral average of the functional.
        numpy.ndarray[float, N]: Standard deviation error associated to each instant of time in the path integral average.
    """
    functional_samples = _generate_functional_samples(functional, S_per_timeslice, N, N_cf, N_cor, eps, thermalization_its)
    if bootstrap:
        matrix_of_functionals_bootstrap = np.zeros((N_cf, N))
        for rows in range(N_cf):
            index_of_copied_path = int(np.random.uniform(0, N_cf))
            for n in range(N):
                matrix_of_functionals_bootstrap[rows][n] = functional_samples[index_of_copied_path][n]
        avg = matrix_of_functionals_bootstrap.mean(axis=0)
        std = matrix_of_functionals_bootstrap.std(axis=0) / np.sqrt(N_cf)
    else:
        avg = functional_samples.mean(axis=0)
        std = functional_samples.std(axis=0) / np.sqrt(N_cf)
    return avg, std
