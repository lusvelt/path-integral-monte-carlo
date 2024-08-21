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
    bin_size: int = 1,
):
    N_bins = int(np.ceil(N_cf / bin_size))
    functional_samples = np.zeros((N_bins, N))
    bin_samples = np.zeros((bin_size, N))
    path = np.zeros(N)
    for _ in range(thermalization_its * N_cor):  # thermalization
        _update_path(path, S_per_timeslice, eps)
    for i in range(N_cf):
        for _ in range(N_cor):  # discard N_cor values
            _update_path(path, S_per_timeslice, eps)
        for n in range(N):  # for every time instant we have N_cf values of G
            bin_samples[i % bin_size][n] = functional(path, n)
        if (i + 1) % bin_size == 0 or i == N_cf - 1:
            functional_samples[i // bin_size] = bin_samples.mean(axis=0)
    return functional_samples


def compute_path_integral_average(
    functional: Callable,
    S_per_timeslice: Callable,
    N: int,
    N_cf: int,
    N_cor: int,
    eps: float,
    thermalization_its: int = 5,
    N_copies: int = 1,
    bin_size: int = 1,
):
    """
    Computes the path integral average

    $$\\langle\\langle \\Gamma[x] \\rangle\\rangle = \\frac{\\int Dx \\Gamma[x] e^{-S[x]}}{\\int Dx e^{-S[x]}}$$

    using the Metropolis Monte Carlo method explained in section 2.2 of "Lattice QCD for Novices" of P. Lepage

    Args:
        functional(Callable[numpy.ndarray[float, N]] -> float): A functional taking a path (as a numpy array of length N) and a (discretized) time instant and returning a number.
        S_per_timeslice(Callable[int, numpy.ndarray[float, N]] -> float): A functional taking as input an integer $j$ and a path, and returning the contribution of the $j$-th point of the path to the action.
        N (int): Number of path points.
        N_cf (int): Total number of samples contributing to be saved during the process for computing the path integral average.
        N_cor (int): Number of path updates before picking each sample.
        eps (float): $\\epsilon$ parameter for the update of the path.
        thermalization_its (int, optional): Number of samples to be discarded at the beginning to let the procedure thermalize. Default is 5.
        N_copies (int, optional): Number of bootstrap averages to be returned. Default is 1 (no bootstrap).
        bin_size (int, optional): Number of samples to be averaged in a single bin.

    Returns:
        numpy.ndarray[float, N_copies * N]: Path integral average of the functional.
        numpy.ndarray[float, N_copies * N]: Standard deviation error associated to each instant of time in the path integral average.
    """
    assert N_copies <= N_cf
    assert bin_size <= N_cf
    functional_samples = _generate_functional_samples(functional, S_per_timeslice, N, N_cf, N_cor, eps, thermalization_its, bin_size)
    N_bins = int(np.ceil(N_cf / bin_size))  # if bin_size == 1, then N_bins == N_cf
    if N_copies > 1:
        bootstrap_avgs = np.zeros((N_copies, N))
        bootstrap_stds = np.zeros((N_copies, N))
        for i in range(N_copies):
            matrix_of_functionals_bootstrap = np.zeros((N_bins, N))
            for rows in range(N_bins):
                index_of_copied_path = int(np.random.uniform(0, N_bins))
                for n in range(N):
                    matrix_of_functionals_bootstrap[rows][n] = functional_samples[index_of_copied_path][n]
            bootstrap_avgs[i] = matrix_of_functionals_bootstrap.mean(axis=0)
            bootstrap_stds[i] = matrix_of_functionals_bootstrap.std(axis=0) / np.sqrt(N_bins)
        return bootstrap_avgs, bootstrap_stds
    else:
        avg = functional_samples.mean(axis=0)
        std = functional_samples.std(axis=0) / np.sqrt(N_bins)
        return avg, std
