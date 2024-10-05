"""
This module provides functions to compute path integrals using Metropolis Monte Carlo method (explained in section 2.3 of Lepage's paper "Lattice QCD for Novices")
"""

import numpy as np
from numba import njit


@njit
def update_path(path, S_per_timeslice, eps: np.float64):
    """
    Metropolis Monte Carlo update of the path explained in section 2.2 of "Lattice QCD for Novices" of P. Lepage

    Args:
        path (numpy.ndarray[float, N]): The path of the lattice to be updated.
        S_per_timeslice(Callable[int, numpy.ndarray[float, N]] -> float): A functional taking as input an integer $j$ and a path, and returning the contribution of the $j$-th point of the path to the action.
        eps (float): $\\epsilon$ parameter for the update of the path.
    """
    N = path.shape[0]
    for j in range(N):
        # save original value
        old_x = path[j]
        old_Sj = S_per_timeslice(j, path)
        # update path[j]
        path[j] += np.random.uniform(-eps, eps)
        # change in action
        dS = S_per_timeslice(j, path) - old_Sj
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            # restore old value
            path[j] = old_x


@njit
def generate_functional_samples(
    functional,
    S_per_timeslice,
    N: np.int32,
    N_cf: np.int32,
    N_cor: np.int32,
    eps: np.float64,
    thermalization_its: np.int32,
    bin_size: np.int32,
    N_points: np.int32,
):
    """
    Computes the matrix of functional samples, where each row is the average of the functional over N_points time instants for a given bin.

    Args:
        functional(Callable[numpy.ndarray[float, N]] -> float): A functional taking a path (as a numpy array of length N) and a (discretized) time instant and returning a number.
        S_per_timeslice(Callable[int, numpy.ndarray[float, N]] -> float): A functional taking as input an integer $j$ and a path, and returning the contribution of the $j$-th point of the path to the action.
        N (int): Number of path points.
        N_cf (int): Total number of samples contributing to be saved during the process for computing the path integral average.
        N_cor (int): Number of path updates before picking each sample.
        eps (float): $\\epsilon$ parameter for the update of the path.
        thermalization_its (int, optional): Number of samples to be discarded at the beginning to let the procedure thermalize. Default is 5.
        bin_size (int, optional): Number of samples to be averaged in a single bin.
        N_points (int, optional): Number of points to be averaged in the path integral average. Default is N.

    Returns:
        numpy.ndarray[float, N_bins * N_points]: Matrix of functional samples.
    """
    N_bins = int(np.ceil(N_cf / bin_size))
    functional_samples = np.zeros((N_bins, N_points), dtype=np.float64)
    bin_samples = np.zeros((bin_size, N_points), dtype=np.float64)
    path = np.zeros(N, np.float64)
    # thermalization
    for _ in range(thermalization_its * N_cor):
        update_path(path, S_per_timeslice, eps)
    for i in range(N_cf):
        # discard N_cor values of G
        for _ in range(N_cor):
            update_path(path, S_per_timeslice, eps)
        # for every time instant we have N_cf values of G
        for n in range(N_points):
            bin_samples[i % bin_size][n] = functional(path, n)
        if (i + 1) % bin_size == 0 or i == N_cf - 1:
            for n in range(N_points):
                functional_samples[i // bin_size, n] = np.sum(bin_samples[:, n]) / bin_size
    return functional_samples


@njit
def compute_path_integral_average(
    functional,
    S_per_timeslice,
    N: np.int32,
    N_cf: np.int32,
    N_cor: np.int32,
    eps: np.float64,
    thermalization_its: np.int32 = 20,
    N_copies: np.int32 = 1,
    bin_size: np.int32 = 1,
    N_points: np.int32 = None,
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
        N_points (int, optional): Number of time instants to be considered in the path integral average. Default is N.

    Returns:
        numpy.ndarray[float, N_copies * N]: N_copies boostrap calculations of the path integral average of the functional.
        numpy.ndarray[float, N_copies * N]: Standard deviation error associated to each instant of time in the path integral average, for each bootstrap.
    """
    assert N_copies <= N_cf
    assert bin_size <= N_cf

    if N_points is None:
        N_points = N

    assert 0 < N_points <= N

    functional_samples = generate_functional_samples(
        functional,
        S_per_timeslice,
        N,
        N_cf,
        N_cor,
        eps,
        thermalization_its,
        bin_size,
        N_points,
    )
    # if bin_size == 1, then N_bins == N_cf
    N_bins = int(np.ceil(N_cf / bin_size))
    avgs = np.zeros((N_copies, N_points), dtype=np.float64)
    stds = np.zeros((N_copies, N_points), dtype=np.float64)
    if N_copies > 1:  # bootstrap procedure, see paper
        for i in range(N_copies):
            # create a matrix of functional samples by picking randomnly N_bins samples the functional_samples matrix
            matrix_of_functionals_bootstrap = np.zeros((N_bins, N_points), dtype=np.float64)
            for rows in range(N_bins):
                index_of_copied_path = np.int32(np.random.uniform(0, N_bins))
                for n in range(N_points):
                    matrix_of_functionals_bootstrap[rows][n] = functional_samples[index_of_copied_path][n]
            for n in range(N_points):
                avgs[i, n] = np.sum(matrix_of_functionals_bootstrap[:, n]) / N_bins
                stds[i, n] = np.sum((matrix_of_functionals_bootstrap[:, n] - avgs[i, n]) ** 2) / (N_bins * np.sqrt(N_bins))
        return avgs, stds
    else:
        for n in range(N_points):
            avgs[0, n] = np.sum(functional_samples[:, n]) / N_bins
            stds[0, n] = np.sum((functional_samples[:, n] - avgs[0, n]) ** 2) / (N_bins * np.sqrt(N_bins))
        return avgs, stds
