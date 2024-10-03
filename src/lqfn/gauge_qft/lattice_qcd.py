"""
This module contains functions to compute quantities in lattice QCD.
"""

import numpy as np
from numba import njit, prange


@njit
def factorial(n):
    """
    Computes the factorial of a given integer.

    Args:
        n (int): The integer for which the factorial is to be computed.

    Returns:
        int: The factorial of the given integer.
    """
    f = 1
    for i in range(1, n + 1):
        f *= i
    return f


@njit
def generate_random_SU3_update_matrix(eps, taylor_order=50):
    """
    Generates a random SU(3) matrix $M$ such that
    $$M = \\exp(i \\epsilon H)$$,
    where $H$ is a random Hermitian matrix with values +1 or -1.

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
    H = (H + H.conj().T) / 2
    for n in range(taylor_order):
        M = M + (1j * eps) ** n / factorial(n) * np.linalg.matrix_power(H, n)
    M = M / np.linalg.det(M) ** (1 / 3)
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
        s[N + i] = s[i].conj().T
    return s


@njit
def decode_index(index, N, d):
    """
    Decodes an index into a multi-index.

    Args:
        index (int): the index to be decoded.
        N (int): the number of lattice sites in each direction.
        d (int): the number of dimensions of the lattice.

    Returns:
        numpy.ndarray[int, d]: The multi-index corresponding to the given index.
    """
    result = np.zeros(d, dtype=np.int32)
    for i in range(d):
        result[i] = index % N
        index = index // N
    return result


@njit
def encode_index(x, N, d):
    """
    Encodes a multi-index into an index.

    Args:
        x(numpy.ndarray[int, d]): the multi-index to be encoded.
        N(int): the number of lattice sites in each direction.
        d(int): the number of dimensions of the lattice.

    Returns:
        int: the index corresponding to the given multi-index.
    """
    result = 0
    for i in range(d):
        result += N**i * x[i]
    return np.int32(result)


@njit
def create_lattice_links(N, d):
    """
    Creates the lattice links.

    Args:
        N(int): the number of lattice sites in each direction.
        d(int): the number of dimensions of the lattice.

    Returns:
        numpy.ndarray[complex, N^d, d, 3, 3]: the lattice links.
    """
    shape = (N**d, d, 3, 3)
    links = np.zeros(shape, dtype=np.complex128)
    for i in range(N**d):
        for mu in range(d):
            links[i][mu] = np.identity(3, dtype=np.complex128)
    return links


# Links are stored in the following way:
# - there are N^d nodes
# - to index a node, we need d indices from 0 to N-1 (e.g. [t, x, y, z])
# - after having indedxed a node, there are d "forward" directions, so we need to index the direction mu of the link we desire
# - the value after these indicizations is a SU(3) matrix, which encodes the field configuration


@njit
def get_node(links, x):
    """
    Returns the node at a given multi-index.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): The lattice links.
        x(numpy.ndarray[int, d]): The multi-index of the node.

    Returns:
        numpy.ndarray[complex, d, 3, 3]: the node at the given multi-index.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    # Returns a numpy array containing d matrices, one for each spacetime direction
    i = encode_index(x, N, d)
    return links[i]


@njit
def get_link(links, x, step):
    """
    Returns the U matrix of a given node along the direction given from the variable step.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): The lattice links.
        x(numpy.ndarray[int, d]): The multi-index of the node.
        step(int): The step to be taken.

    Returns:
        numpy.ndarray[complex, 3, 3]: The U matrix.
    """
    # Returns the U matrix
    # step is an integer from 1 to d
    # if the step is +, then it's forward in that direction, if it's - then it's backwards
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    assert step != 0
    assert np.abs(step) <= d
    mu = np.abs(step) - 1
    x = np.copy(x)
    if step > 0:
        node = get_node(links, x)
        link = node[mu]
    else:
        x[mu] -= 1
        # PBC
        if x[mu] < 0:
            x[mu] += N

        node = get_node(links, x)
        link = node[mu].conj().T
    return link


@njit
def compute_path(links, x_start, steps):
    """
    Computes the path along a given set of steps starting from a given node and link.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): The lattice links.
        x_start(numpy.ndarray[int, d]): the starting node.
        steps(numpy.ndarray[int, n]): The steps to be taken.

    Returns:
        numpy.ndarray[complex, 3, 3]: the path along the given steps.
    """
    # start_x is a lattice multi_index
    # steps is an array of steps: each step is +/- mu, where mu is a spacetime index from 1 to d
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    assert x_start.shape == (d,)
    Us = np.identity(3, dtype=np.complex128)
    x = np.copy(x_start)
    for step in steps:
        if step != 0:
            Us = Us @ get_link(links, x, step)
            mu = np.abs(step) - 1
            x[mu] += np.sign(step)
            # Implement periodic boundary conditions
            if x[mu] >= N:
                x[mu] -= N
            elif x[mu] < 0:
                x[mu] += N
    return Us


@njit
def compute_plaquette(links, mu, nu, x):
    """
    Computes the plaquette at a given node and direction.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        mu(int): the first direction.
        nu(int): the second direction.
        x(numpy.ndarray[int, d]): the multi-index of the node.

    Returns:
        float: the value of the plaquette.
    """
    s1 = mu + 1
    s2 = nu + 1
    steps = [s1, s2, -s1, -s2]
    path = compute_path(links, x, steps)
    return 1 / 3 * np.real(np.trace(path))


@njit
def compute_wilson_action(links, beta):
    """
    Computes the non improve Wilson action.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        beta(float): the beta parameter that enters in the calculation of the Wilson action.

    Returns:
        float: the value of the Wilson action.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    P = 0
    for i in range(N**d):
        x = decode_index(i, N, d)
        for mu in range(d):
            for nu in range(mu):
                P += compute_plaquette(links, mu, nu, x)
    return -beta * P


@njit
def compute_gamma(links, x, mu):
    """
    Computes the Gamma matrix at a given node and direction.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        x(numpy.ndarray[int, d]): the multi-index of the node.
        mu(int): the direction.

    Returns:
        numpy.ndarray[complex, 3, 3]: the Gamma matrix.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    # Computes the Gamma (113) in the paper, given a link specified as start point and direction mu
    gamma = np.zeros((3, 3), dtype=np.complex128)
    x[mu] += 1
    if x[mu] >= N:
        x[mu] -= N
    # For each direction there are two plaquettes, except for the direction of the link itself
    for nu in range(d):
        if nu != mu:
            # remaining path
            s1 = mu + 1
            s2 = nu + 1
            path_forward = compute_path(links, x, np.array([s2, -s1, -s2], dtype=np.int32))
            path_backward = compute_path(links, x, np.array([-s2, -s1, s2], dtype=np.int32))
            gamma += path_forward + path_backward
    return gamma


@njit
def compute_gamma_improved(links, x, mu, u0):
    """
    Computes the improve Gamma matrix at a given node and direction, meaning that there are both plaquettes and rectangles inside gamma.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        x(numpy.ndarray[int, d]): the multi-index of the node.
        mu(int): the direction.

    Returns:
        numpy.ndarray[complex, 3, 3]: the Gamma matrix.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    # Computes the Gamma (113) in the paper, given a link specified as start point and direction mu
    gamma = np.zeros((3, 3), dtype=np.complex128)
    x[mu] += 1
    if x[mu] >= N:
        x[mu] -= N
    # there are 8 rectangles for each direction
    for nu in range(d):
        if nu != mu:
            # remaining path
            s1 = mu + 1
            s2 = nu + 1
            plaquette_path_forward = compute_path(links, x, np.array([s2, -s1, -s2], dtype=np.int32))
            plaquette_path_backward = compute_path(links, x, np.array([-s2, -s1, s2], dtype=np.int32))

            rectangles_steps = np.array(
                [
                    [s1, s2, -s1, -s1, -s2],
                    [s1, -s2, -s1, -s1, s2],
                    [s2, -s1, -s1, -s2, s1],
                    [-s2, -s1, -s1, s2, s1],
                    [s2, s2, -s1, -s2, -s2],
                    [-s2, -s2, -s1, s2, s2],
                ],
                dtype=np.int32,
            )

            rectangle_paths = np.zeros((6, 3, 3), dtype=np.complex128)
            for i in range(rectangles_steps.shape[0]):
                rectangle_paths[i] = compute_path(links, x, rectangles_steps[i])
            rectangle_contributions = np.zeros((3, 3), dtype=np.complex128)
            for i in range(rectangle_paths.shape[0]):
                rectangle_contributions += rectangle_paths[i]
            plaquette_contributions = plaquette_path_forward + plaquette_path_backward
            gamma += 5 / 3 * 1 / u0**4 * plaquette_contributions - 1 / u0**6 * 1 / 12 * rectangle_contributions
    return gamma


@njit
def compute_action_contribution(links, x, mu, gamma, beta):
    """
    Computes the action contribution at a given node and direction given a Gamma matrix.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        x(numpy.ndarray[int, d]): the multi-index of the node.
        mu(int): the direction.
        gamma(numpy.ndarray[complex, 3, 3]): the Gamma matrix.
        beta(float): the beta parameter that enters in the calculation of the Wilson action.

    Returns:
        float: the value of the action contribution.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    assert x.shape == (d,)
    assert 0 <= mu < d
    U = get_link(links, x, mu + 1)
    return -beta / 3 * np.real(np.trace(U @ gamma))


@njit
def pick_random_matrix(random_matrices):
    """
    Picks a random matrix from a given set of matrices.

    Args:
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices.

    Returns:
        numpy.ndarray[complex, 3, 3]: the picked random matrix.
    """
    i = np.random.choice(random_matrices.shape[0])
    return random_matrices[i]


@njit
def update_link(links, x, mu, hits, beta, random_matrices, u0, improved):
    """
    Updates a link at a given node and direction. Each link is updated hits number of times.

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
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    U = get_node(links, x)[mu]
    i = encode_index(x, N, d)
    if improved is True:
        gamma = compute_gamma_improved(links, x, mu, u0)
    else:
        gamma = compute_gamma(links, x, mu)
    for _ in range(hits):
        old_U = np.copy(U)
        M = pick_random_matrix(random_matrices)
        links[i][mu] = M @ links[i][mu]
        dS = -beta / 3 * np.real(np.trace((links[i][mu] - old_U) @ gamma))
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            links[i][mu] = old_U


@njit
def update_lattice(links, hits, beta, random_matrices, u0, improved):
    """
    Updates the lattice links.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        hits(int): the number of updates.
        beta(float): the beta parameter that enters in the calculation of the Wilson action.
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices.
        u0(float): the u0 parameter that enters in the calculation of the Wilson action.
        improved(bool): whether to use the improved Gamma matrix or not.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    for i in range(N**d):
        for mu in range(d):
            update_link(links, decode_index(i, N, d), mu, hits, beta, random_matrices, u0, improved)


@njit
def generate_wilson_samples(links, steps, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved):
    """
    Computes the metropolis algorithm to calculate Wilson loop average and its error.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        steps(numpy.ndarray[int, n, d]): the steps to be taken.
        N_cf(int): the total number of samples contributing to be saved during the process for computing the path integral average.
        N_cor(int): the number of path updates before picking each sample.
        hits(int): the number of updates.
        thermalization_its(int): the number of samples to be discarded at the beginning to let the procedure thermalize.
        bin_size(int): the number of samples to be averaged in a single bin.
        beta(float): the beta parameter that enters in the calculation of the Wilson action.
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices.
        u0(float): the u0 parameter that enters in the calculation of the Wilson action.
        improved(bool): whether to use the improved Gamma matrix or not.

    Returns:
        numpy.ndarray[float, N_bins, steps.shape[0]]: the matrix of Wilson loop samples.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    n_loops_to_compute = steps.shape[0]

    N_bins = int(np.ceil(N_cf / bin_size))
    wilson_samples = np.zeros((N_bins, n_loops_to_compute), dtype=np.float64)
    bin_samples = np.zeros((bin_size, n_loops_to_compute), dtype=np.float64)
    for i in range(thermalization_its):  # thermalization
        print(f"{i}/{thermalization_its} thermalization iteration")
        for _ in range(N_cor):
            update_lattice(links, hits, beta, random_matrices, u0, improved)
    for i in range(N_cf):
        print(f"{i}/{N_cf}")
        for _ in range(N_cor):  # discard N_cor values
            update_lattice(links, hits, beta, random_matrices, u0, improved)
        for j in range(n_loops_to_compute):
            # sweep through all possible loops of the current kind in the lattice
            value = 0
            for k in range(N**d):
                y = decode_index(k, N, d)
                path = compute_path(links, y, steps[j, :])
                value += 1 / 3 * np.real(np.trace(path))
            value /= N**d
            bin_samples[i % bin_size][j] = value
        if (i + 1) % bin_size == 0 or i == N_cf - 1:
            for j in range(n_loops_to_compute):
                wilson_samples[i // bin_size][j] = bin_samples[:, j].mean()
                print(wilson_samples[i // bin_size][j])
    return wilson_samples


@njit(parallel=True)
def compute_path_integral_average(links, steps, N_cf, N_cor, hits, thermalization_its, N_copies, bin_size, beta, random_matrices, u0, improved):
    """
    Computes the metropolis algorithm to calculate Wilson loop average and its error; if N_copies is greater than one than it uses bootstrap procedure.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        steps(numpy.ndarray[int, n, d]): the steps to be taken.
        N_cf(int): the total number of samples contributing to be saved during the process for computing the path integral average.
        N_cor(int): the number of path updates before picking each sample.
        hits(int): the number of updates.
        thermalization_its(int): the number of samples to be discarded at the beginning to let the procedure thermalize.
        N_copies(int): the number of copies for the bootstrap procedure.
        bin_size(int): the number of samples to be averaged in a single bin.
        beta(float): the beta parameter that enters in the calculation of the Wilson action.
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices.
        u0(float): the u0 parameter that enters in the calculation of the Wilson action.
        improved(bool): whether to use the improved Gamma matrix or not.

    Returns:
        float: average of the Wilson loops.
        float: error of the average of the Wilson loops.
    """

    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    assert N_copies <= N_cf
    assert bin_size <= N_cf

    n_loops_to_compute = steps.shape[0]

    wilson_samples = generate_wilson_samples(links, steps, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved)
    N_bins = int(np.ceil(N_cf / bin_size))  # if bin_size == 1, then N_bins == N_cf
    # bootstrap procedure
    if N_copies > 1:
        bootstrap_avgs = np.zeros((N_copies, n_loops_to_compute), dtype=np.float64)
        for i in range(N_copies):
            values = np.zeros((N_bins, n_loops_to_compute), dtype=np.float64)
            for j in range(N_bins):
                index_of_copied_value = int(np.random.uniform(0, N_bins))
                for k in range(n_loops_to_compute):
                    values[j, k] = wilson_samples[index_of_copied_value, k]
            for k in range(n_loops_to_compute):
                bootstrap_avgs[i, k] = values[:, k].mean()
        wilson_samples = bootstrap_avgs
    return wilson_samples


@njit
def get_steps_for_rectangle(width, height, mu, nu):
    """
    Returns the steps for a rectangle.

    Args:
        width(int): the width of the rectangle.
        height(int): the height of the rectangle.
        mu(int): the first direction.
        nu(int): the second direction.

    Returns:
        numpy.ndarray[int, 2 * (width + height)]: the steps for the rectangle.
    """
    assert 1 <= width
    assert 1 <= height
    assert 0 <= mu
    assert 0 <= nu
    s1 = mu + 1
    s2 = nu + 1
    length = 2 * (width + height)
    steps = np.zeros(length, dtype=np.int32)
    for i in range(width):
        steps[i] = s1
    for i in range(width, width + height):
        steps[i] = s2
    for i in range(width + height, 2 * width + height):
        steps[i] = -s1
    for i in range(2 * width + height, length):
        steps[i] = -s2
    return steps


# computes a^2 \Delta^2 U
@njit
def compute_gauge_covariant_derivative(links, x, mu, u0):
    """
    Computes the gauge covariant derivative at a given node and direction.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        x(numpy.ndarray[int, d]): the multi-index of the node.
        mu(int): the direction.
        u0(float): the u0 parameter that enters in the calculation of the Wilson action.

    Returns:
        numpy.ndarray[complex, 3, 3]: the gauge covariant derivative.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    D = np.zeros((3, 3), dtype=np.complex128)
    for rho in range(d):
        s1 = mu + 1
        s2 = rho + 1
        path1 = compute_path(links, x, np.array([s2, s1, -s2]))
        path2 = compute_path(links, x, np.array([-s2, s1, s2]))
        path3 = -2 * u0**2 * get_link(links, x, s1)
        D += path1 + path2 + path3
    return D / u0**2


@njit
def smear_matrix(links, x, mu, u0, eps, n):
    """
    Calcuate the smears a matrix at a given node and direction.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        x(numpy.ndarray[int, d]): the multi-index of the node.
        mu(int): the direction.
        u0(float): the u0 parameter that enters in the calculation of the Wilson action.
        eps(float): the epsilon parameter for the calculation of the smeared matrix.
        n(int): the number of updates.

    Returns:
        numpy.ndarray[complex, 3, 3]: the smeared matrix
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    i = encode_index(x, N, d)
    links[i][mu] = np.identity(3) + eps * compute_gauge_covariant_derivative(links, x, mu, u0)
    if n > 1:
        smear_matrix(links, x, mu, u0, eps, n - 1)


@njit
def smear_links(links, mu, u0, eps, n):
    """
    Smears all the links in a given direction.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links.
        mu(int): the direction.
        u0(float): the u0 parameter that enters in the calculation of the Wilson action.
        eps(float): the epsilon parameter for the calculation of the smeared matrix.
        n(int): the number of updates.

    Returns:
        numpy.ndarray[complex, N^d, d, 3, 3]: the smeared links.

    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    old_links = np.copy(links)
    new_links = np.zeros_like(links)
    for i in range(links.shape[0]):
        x = decode_index(i, N, d)
        links = np.copy(old_links)
        smear_matrix(links, x, mu, u0, eps, n)
        new_links[i, mu] = np.copy(links[i, mu])


# perchè c'è una funzione dentro una funzione??(i punti interrogativi me li ha suggeriti copilot)
@njit(parallel=True)
def compute_static_quark_potential(
    N, d, N_cf, N_cor, hits, thermalization_its, N_copies, bin_size, beta, random_matrices, u0, improved, eps_smearing=0.0, n_smearing=0
):
    links = create_lattice_links(N, d)
    if eps_smearing != 0.0 and n_smearing != 0:
        # Smear all spatial directions
        for mu in range(1, d):
            smear_links(links, mu, u0, eps_smearing, n_smearing)
    rs = np.arange(1, N)
    results = np.zeros((N - 1, N_copies, 2), dtype=np.float64)

    def compute_single_value(i):
        r = rs[i]
        l = np.copy(links)
        loops = np.zeros((2, 2 * (N - 1 + r)), dtype=np.int32)
        # t loop
        steps_t = get_steps_for_rectangle(N - 2, r, 0, 1)
        for j in prange(steps_t.shape[0]):
            loops[0, j] = steps_t[j]
        # t+a loop
        loops[1] = get_steps_for_rectangle(N - 1, r, 0, 1)
        print(loops)
        result = compute_path_integral_average(
            l, loops, N_cf, N_cor, hits, thermalization_its, N_copies, bin_size, beta, random_matrices, u0, improved
        )
        return result

    for i in range(N - 1):
        results[i] = compute_single_value(i)

    V_bootstrap = np.zeros((N_copies, N - 1), dtype=np.float64)
    for i in range(N_copies):
        for j in range(N - 1):
            V_bootstrap[i, j] = np.log(np.abs(results[j, i, 0] / results[j, i, 1]))
    return V_bootstrap
