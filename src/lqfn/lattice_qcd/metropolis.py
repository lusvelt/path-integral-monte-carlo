"""
This module contains functions to compute quantities in lattice QCD using the Metropolis Monte Carlo procedure.
Every function is decorated with numba's `njit`, since the tasks are computationally expensive.
"""

from numba import njit
from .lattice import *
from ..combinatorics import *


@njit
def generate_random_SU3_update_matrix(eps, taylor_order=50):
    """
    Generates a random SU(3) matrix $M$ such that
    $$M = \\exp(i \\epsilon H)$$,
    where $H$ is a random hermitian matrix with complex values having real and imaginary part between -1 and 1.

    Args:
        eps (float): $\\epsilon$ parameter for the update of the matrix.
        taylor_order (int, optional): order of the Taylor expansion to compute the exponential. Default is 50.

    Returns:
        numpy.ndarray[complex, 3, 3]: The generated SU(3) matrix.
    """
    assert 0 < eps < 1
    M = np.zeros((3, 3), dtype=np.complex128)
    H = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            H[i, j] = complex(np.random.uniform(-1, 1), np.random.uniform(-1, 1))
    H = (H + H.conj().T) / 2  # Hermiticize H
    # Taylor expansion of M
    for n in range(taylor_order):
        M = M + (1j * eps) ** n / factorial(n) * np.linalg.matrix_power(H, n)
    M = M / np.linalg.det(M) ** (1 / 3)  # Normalize the determinant to 1
    return M


@njit
def generate_update_matrices_set(N, eps):
    """
    Assign an update matrix for every lattice link.

    Args:
        N (int): number of lattice sites in each direction.
        eps (float): $\\epsilon$ parameter for the update of the matrix.

    Returns:
        numpy.ndarray[complex, N * 2, 3, 3]: The generated set of update matrices.
    """
    assert 0 < eps < 1
    s = np.zeros((N * 2, 3, 3), dtype=np.complex128)
    for i in range(N):
        s[i] = generate_random_SU3_update_matrix(eps)
        s[N + i] = s[i].conj().T  # Insert also the hermitian conjugate of the generated matrix in the set
    return s


@njit
def update_link(links, x, mu, hits, beta, random_matrices, u0, improved):
    """
    Updates a link at a given node and direction. Each link is updated `hits` number of times.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        x(numpy.ndarray[int, d]): the multi-index of the node.
        mu(int): the direction.
        hits(int): the number of updates.
        beta(float): the beta parameter that enters in the calculation of the Wilson action.
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices.
        u0(float): the u0 parameter that enters in the calculation of the Wilson action.
        improved(bool): whether to use the improved Gamma matrix or not.
    """
    d, N = get_d_N(links)

    U = get_node(links, x)[mu]
    i = encode_index(x, N, d)
    if improved is True:
        gamma = compute_gamma_improved(links, x, mu, u0)
    else:
        gamma = compute_gamma(links, x, mu)
    for _ in range(hits):
        old_U = np.copy(U)
        # update U -> MU
        M = pick_random_matrix(random_matrices)
        links[i][mu] = M @ links[i][mu]
        # compute the action change after the update
        # since it is linear, we can compute Delta S by inserting (MU - U) in place of U
        dS = -beta / 3 * np.real(np.trace((links[i][mu] - old_U) @ gamma))
        # check Metropolis acceptance condition, and if it fails restore the previous link
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            links[i][mu] = old_U


@njit
def update_lattice(links, hits, beta, random_matrices, u0, improved):
    """
    Perform a sweep through the lattice of link updates.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        hits(int): the number of updates
        beta(float): the beta parameter that enters in the calculation of the Wilson action
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices
        u0(float): the u0 parameter that enters in the calculation of the Wilson action
        improved(bool): whether to use the improved Gamma matrix or not
    """
    d, N = get_d_N(links)
    # for each spacetime point and for each direction, there is a U matrix to update
    for i in range(N**d):
        for mu in range(d):
            update_link(links, decode_index(i, N, d), mu, hits, beta, random_matrices, u0, improved)


def compute_wilson_loop_average_fixed_frame(links, loop, rotated_axes):
    """
    Computes the average of all the Wilson loops of the specified kind in a fixed rotational frame, just sweeping through all the spacetime points.

    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        loop (numpy.ndarray[int]): the steps of the loop
        rotated_axis (numpy.ndarray[int]): a permutation of the first d integers, representing the rotated frame.
            If the element of index `mu` has value `nu`, it means that the old axis `mu` has been rotated into the new axis `nu`.

    Returns:
        float: the value of the Wilson loop partial average
    """
    d, N = get_d_N(links)
    sum_loops = 0  # this will contain the value for the current rotational configuration, after sweeping through all spacetime points
    rotated_loop = rotate_loop(loop, rotated_axes)
    # sweep through all spacetime points and add up the loop values
    for k in range(N**d):
        y = decode_index(k, N, d)
        path = compute_path(links, y, rotated_loop)
        sum_loops += 1 / 3 * np.real(np.trace(path))
    return sum_loops / (N**d)


@njit
def generate_wilson_samples(links, loops, beta, u0, random_matrices, N_cf, N_cor, hits=10, thermalization_its=2, improved=False, rotate_time=True):
    """
    Performs the metropolis algorithm to generate samples that will contribute to the path integral average.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        loops(numpy.ndarray[int, n_loops, max_loop_length]): the list of loops to compute
        beta(float): the beta parameter that enters in the calculation of the Wilson action
        u0(float): the u0 parameter that enters in the calculation of the Wilson action
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices from which the update matrices are drawn
        N_cf(int): the total number of samples to be generated
        N_cor(int): the number of path updates before picking each sample
        hits(int): the number of updates of each link before going to the next. Default is 10.
        thermalization_its(int): the number of times that N_cor samples are discarded at the beginning to let the procedure thermalize. Default is 2.
        improved(bool): whether to use the improved action or not. Default is False.
        rotate_time(bool): whether or not to rotate time dimension when exploiting rotational symmetry of the lattice (if spatial directions have been smeared, set this to False). Default is True.

    Returns:
        numpy.ndarray[float, N_bins, steps.shape[0]]: the matrix of Wilson loop samples
    """
    # detect the spacetime dimensions d and the number of lattice points N from the shape of the links data structure
    d, N = get_d_N(links)

    n_loops_to_compute = loops.shape[0]

    # prepare the arrays of samples. There is an additional index because there are more loops to compute and save.
    wilson_samples = np.zeros((N_cf, n_loops_to_compute), dtype=np.float64)

    # thermalization updates
    for i in range(thermalization_its * N_cor):
        update_lattice(links, hits, beta, random_matrices, u0, improved)

    # to exploit the discrete rotational symmetry of the lattice and generate more Wilson loop samples,
    # compute all permutations of the values of spacetime indices (0, ..., d-1)
    directions_permutations = permute(np.arange(d))
    rot_factor = factorial(d)
    # if time is not to be rotated, just use the permutations where the time index is first
    if rotate_time is False:
        rot_factor = factorial(d - 1)

    # generate N_gf samples
    for i in range(N_cf):
        # perform N_cor updates to thermalize
        for _ in range(N_cor):
            update_lattice(links, hits, beta, random_matrices, u0, improved)

        # there are more loops to compute
        for j in range(n_loops_to_compute):
            # sweep through all possible loops of the current kind in the lattice, also exploiting rotational symmetry
            value = 0  # this will contain the sum of all the contributions for each rotational configurations
            loop = loops[j, :]
            # for each rotational configuration, sweep through all spacetime points and compute the loop
            for l in range(rot_factor):
                # average the loops value and add them to the total
                value += compute_wilson_loop_average_fixed_frame(links, loop, directions_permutations[l])
            # average the contributions given by each rotational configuration and save them in the bin
            wilson_samples[i, j] = value / rot_factor
    return wilson_samples


@njit(parallel=True)
def compute_wilson_loop_average(
    links, loops, beta, u0, random_matrices, N_cf, N_cor, N_copies, hits=10, thermalization_its=2, improved=False, rotate_time=True
):
    """
    Uses the metropolis algorithm to calculate Wilson loop averages and errors

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        loops(numpy.ndarray[int, n, d]): the list of loops to compute
        beta(float): the beta parameter that enters in the calculation of the Wilson action
        u0(float): the u0 parameter that enters in the calculation of the Wilson action
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices to use for the lattice update
        N_cf(int): the total number of samples to be generated
        N_cor(int): the number of path updates before picking each sample
        N_copies(int): the number of copies for the bootstrap procedure
        hits(int): the number of updates of each link before going to the next. Default is 10
        thermalization_its(int): the number of times that N_cor samples are discarded at the beginning to let the procedure thermalize. Default is 2.
        improved(bool): whether to use the improved action or not. Default is False
        rotate_time(bool): whether or not to rotate time dimension when exploiting rotational symmetry of the lattice (if spatial directions have been smeared, set this to False). Default is True
    Returns:
        np.ndarray[float, N_bins, steps.shape[0]]: the matrix of Wilson loop samples
    """
    n_loops_to_compute = loops.shape[0]

    # generate an array of samples
    wilson_samples = generate_wilson_samples(links, loops, beta, u0, random_matrices, N_cf, N_cor, hits, thermalization_its, improved, rotate_time)

    # bootstrap procedure
    if N_copies > 1:
        # prepare array for output
        bootstrap_avgs = np.zeros((N_copies, n_loops_to_compute), dtype=np.float64)
        # we want to generate N_copies bootstraps
        for i in range(N_copies):
            # prepare array for bootstrap copy
            values = np.zeros((N_cf, n_loops_to_compute), dtype=np.float64)
            # draw N_cf random samples from wilson_samples and save them into 'values'
            for j in range(N_cf):
                index_of_copied_value = int(np.random.uniform(0, N_cf))
                for k in range(n_loops_to_compute):
                    values[j, k] = wilson_samples[index_of_copied_value, k]
            # average and save the result
            for k in range(n_loops_to_compute):
                bootstrap_avgs[i, k] = values[:, k].mean()
        return bootstrap_avgs
    else:
        # if N_copies is 1, no bootstrap
        return wilson_samples


@njit(parallel=True)
def compute_static_quark_potential(
    N,
    d,
    beta,
    u0,
    random_matrices,
    N_cf,
    N_cor,
    N_copies,
    width_t,
    max_r,
    eps_smearing=0.0,
    n_smearing=0,
    hits=10,
    thermalization_its=2,
    improved=False,
):
    """
    Computes the static quark-antiquark potential with lattice QCD using metropolis algorithm, as described in sec. 4.4 of the paper.

    Args:
        N (int): the number of lattice points for each spacetime dimension
        d (int): the number of spacetime dimensions (1, d-1)
        beta(float): the beta parameter that enters in the calculation of the Wilson action
        u0(float): the u0 parameter that enters in the calculation of the Wilson action
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices to use for the lattice update
        N_cf(int): the total number of samples to be generated
        N_cor(int): the number of path updates before picking each sample
        N_copies(int): the number of copies for the bootstrap procedure
        width_t (int): the length of the Wilson loops used for computing the potential, in the temporal direction
        max_r (float): the maximum value of spatial separation between quark-antiquark to be computed
        eps_smearing (float): the $\\epsilon$ parameter for the smearing procedure. Leave unspecified for no smearing.
        n_smearing (int): the number of smearing steps to apply. Leave unspecified (or 0) for no smearing.
        hits(int): the number of updates of each link before going to the next. Default is 10.
        thermalization_its(int): the number of times that N_cor samples are discarded at the beginning to let the procedure thermalize. Default is 2
        improved(bool): whether to use the improved action or not. Default is False
    """
    # initialize the lattice
    links = create_lattice_links(N, d)

    # thermalization updates
    for _ in range(thermalization_its * N_cor):
        update_lattice(links, hits, beta, random_matrices, u0, improved)

    # smearing
    if eps_smearing != 0.0 and n_smearing != 0:
        # smear all spatial directions
        mus = np.arange(1, d)
        smear_links(links, mus, u0, eps_smearing, n_smearing)

    # for each spatial separation we need two loops with t and t+a temporal separation
    width_t_a = width_t + 1

    # compute the length for the longest possible loop
    max_length_loop = (N - 1) * d * 2

    # prepare loops for all spatial separations of the two quarks (up to r_max)
    accepted_separations = get_spatial_separations(N, d, max_r)
    num_separations = accepted_separations.shape[0]

    # for each spatial separation, there are two loops to compute, one with t and the other with (t+a) temporal separation
    loops = np.zeros((num_separations * 2, max_length_loop), dtype=np.int32)

    # prepare the coordinates for the spatial separation, fixing the time separations
    x_t = np.zeros(d, dtype=np.int32)
    x_t_a = np.zeros(d, dtype=np.int32)
    x_t[0] = width_t
    x_t_a[0] = width_t_a

    # for each spatial separation, write down the steps for the two loops
    for i in range(num_separations):
        x = accepted_separations[i]
        # fill x_t and x_t_a in the spatial indices
        x_t[1:] = x
        x_t_a[1:] = x
        # compute the steps sequence to reach the two points in spacetime
        steps_t = get_nonplanar_steps(x_t)
        steps_t_a = get_nonplanar_steps(x_t_a)
        # insert the loops in the full loop list
        loops[i] = steps_t
        loops[i + num_separations] = steps_t_a

    # compute path integral averages for all loops
    results = compute_wilson_loop_average(
        links,
        loops,
        beta,
        u0,
        random_matrices,
        N_cf,
        N_cor,
        N_copies,
        hits,
        thermalization_its=0,  # the lattice is already thermalized
        improved=improved,
        rotate_time=False,
    )

    # create bootstrap copies of the potential
    V_bootstrap = np.zeros((N_copies, num_separations), dtype=np.float64)
    for i in range(N_copies):
        for j in range(num_separations):
            V_bootstrap[i, j] = np.log(np.abs(results[i, j] / results[i, j + num_separations]))

    # arrange the output in a single data structure
    # [0]: r, [1]: V, [2]: err
    return_data = np.zeros((3, num_separations), dtype=np.float64)
    for i in range(num_separations):
        # compute r as the norm of the spatial separation vector
        x = accepted_separations[i]
        r = np.sqrt(np.sum(x**2))
        # average the bootstrapped values for the given spatial separations
        V = np.sum(V_bootstrap[:, i]) / N_copies
        # compute the standard deviation of the bootstrapped values
        err = np.sqrt(np.sum((V_bootstrap[:, i] - V) ** 2) / N_copies)
        # insert results in the output data structure
        return_data[0, i] = r
        return_data[1, i] = V
        return_data[2, i] = err
    return return_data
