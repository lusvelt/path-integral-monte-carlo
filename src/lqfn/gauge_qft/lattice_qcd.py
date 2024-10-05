"""
This module contains functions to compute quantities in lattice QCD.
"""

import numpy as np
from numba import njit
from .combinatorics import *


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
    # prepare set
    s = np.zeros((N * 2, 3, 3), dtype=np.complex128)
    for i in range(N):
        s[i] = generate_random_SU3_update_matrix(eps)
        s[N + i] = s[i].conj().T  # Insert also the hermitian conjugate of the generated matrix in the set
    return s


@njit
def decode_index(index, N, d):
    """
    Converts an integer into a spacetime coordinate array.
    The conversion is such that index is in decimal system and the spacetime coordinates are in base $N$ system, filled with zeros before non-zero digits up to $d$ digits.
    This function is needed because numba does not allow `itertools` to be used.

    Args:
        index (int): the index to be decoded.
        N (int): the number of lattice sites in each direction.
        d (int): the number of dimensions of the lattice.

    Returns:
        numpy.ndarray[int, d]: The spacetime point coordinates corresponding to the given index.
    """
    result = np.zeros(d, dtype=np.int32)
    for i in range(d):
        result[i] = index % N
        index = index // N
    return result


@njit
def encode_index(x, N, d):
    """
    Converts the spacetime point coordinate array into an integer.
    The conversion is such that index is in decimal system and the spacetime coordinates are in base $N$ system, filled with zeros before non-zero digits up to $d$ digits.
    This function is needed because numba does not allow `itertools` to be used.

    Args:
        x(numpy.ndarray[int, d]): the spacetime coordinates to be encoded.
        N(int): the number of lattice sites in each direction.
        d(int): the number of dimensions of the lattice.

    Returns:
        int: the index corresponding to the given spacetime coordinates.
    """
    result = 0
    for i in range(d):
        result += N**i * x[i]
    return np.int32(result)


@njit
def create_lattice_links(N, d):
    """
    Creates the lattice links encoding the field configuration.
    Each link is initialized to be the identity.
    The shape of the links array is $(N^d, d, 3, 3)$:
    - The first index is for the spacetime point $x$, encoded in a decimal integer
    - The second index is for the spacetime direction
    - The other two indices are for the SU(3) matrix

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
        links (numpy.ndarray[complex, N^d, d, 3, 3]): The lattice links data structure
        x (numpy.ndarray[int, d]): The spacetime coordinates of the node

    Returns:
        numpy.ndarray[complex, d, 3, 3]: a numpy array containing d matrices, one for each spacetime direction
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    i = encode_index(x, N, d)
    return links[i]


@njit
def get_link(links, x, step):
    """
    Returns the U matrix (or U dagger) of a given node along the direction of the step

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): The lattice links data structure
        x(numpy.ndarray[int, d]): The spacetime coordinates of the point
        step(int): The step to be taken. It is an integer from 1 to d. If the step is positive, then it's forward in that direction, if it's negative then it's backwards ($U^{\\dagger}$).

    Returns:
        numpy.ndarray[complex, 3, 3]: The U matrix.
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    assert step != 0
    assert np.abs(step) <= d
    # The spacetime index is from 0 to d-1, while the step is from 1 to d with sign
    mu = np.abs(step) - 1
    x = np.copy(x)
    if step > 0:  # the step is in positive direction
        node = get_node(links, x)
        link = node[mu]
    else:  # the step in in negative direction
        # go to the previous node in that direction
        x[mu] -= 1
        # Periodic Boundary Condition
        if x[mu] < 0:
            x[mu] += N
        node = get_node(links, x)
        # take the hermitian conjugate
        link = node[mu].conj().T
    return link


@njit
def compute_path(links, x_start, steps):
    """
    Computes the path along a given set of steps starting from a given node.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        x_start(numpy.ndarray[int, d]): the starting node
        steps(numpy.ndarray[int, n]): the steps to be taken, in step notation (an array of steps, which are integers from 1 to d with sign)

    Returns:
        numpy.ndarray[complex, 3, 3]: the matrix multiplication of all U matrices of the traversed links
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    assert x_start.shape == (d,)
    Us = np.identity(3, dtype=np.complex128)  # initialize variable that will contain the matrices products of the links
    x = np.copy(x_start)

    for step in steps:
        if step != 0:
            # perform the update
            Us = Us @ get_link(links, x, step)
            # the spacetime direction is from 0 to d-1
            mu = np.abs(step) - 1
            # move to the next node
            x[mu] += np.sign(step)
            # implement periodic boundary conditions
            if x[mu] >= N:
                x[mu] -= N
            elif x[mu] < 0:
                x[mu] += N
    return Us


@njit
def compute_plaquette(links, mu, nu, x):
    """
    Computes the plaquette $P_{\\mu\\nu}(x)$ in the current field configuration, as in eq. (88).

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        mu(int): the first direction of the plaquette
        nu(int): the second direction of the plaquette
        x(numpy.ndarray[int, d]): the spacetime coordinates of the point

    Returns:
        float: the value of the plaquette.
    """
    # the spacetime direction is from 0 to d-1, so we need to convert to step notation (from 1 to d)
    s1 = mu + 1
    s2 = nu + 1
    # define the steps of the plaquette (1 forward in mu direction, 1 forward in nu, 1 backwards in mu, 1 backwards in nu)
    steps = np.array([s1, s2, -s1, -s2], dtype=np.int32)
    # implement formula (88)
    path = compute_path(links, x, steps)
    return 1 / 3 * np.real(np.trace(path))


@njit
def compute_wilson_action(links, beta):
    """
    Computes the non improved Wilson action of eq. (114).

    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        beta (float): the beta parameter that enters in the calculation of the Wilson action (recall that $\\beta = \\tilde{\\beta}/u_0^4$)

    Returns:
        float: the value of the Wilson action
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
    Computes the contribution of the neighbours of the link $U_{\\mu}(x)$ to the Wilson action, yet to be multiplied on the left by $U_{\\mu}(x)$ (see eq. (113)).

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        x(numpy.ndarray[int, d]): the spacetime coordinates array of the factorized link
        mu(int): the direction of the factorized link

    Returns:
        numpy.ndarray[complex, 3, 3]: the matrix $\\Gamma_{\\mu}(x)$
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    gamma = np.zeros((3, 3), dtype=np.complex128)
    # move to the end of the link U_mu(x)
    x[mu] += 1
    # periodic boundary conditions
    if x[mu] >= N:
        x[mu] -= N
    # for each direction there are two plaquettes, except for the direction of the link itself
    for nu in range(d):
        if nu != mu:
            # convert the spacetime index (from 0 to d-1) to steps (from 1 to d)
            s1 = mu + 1
            s2 = nu + 1
            # compute the remaining path, starting from the end of the link to its start, tracing three sides of the plaquette
            path_forward = compute_path(links, x, np.array([s2, -s1, -s2], dtype=np.int32))
            path_backward = compute_path(links, x, np.array([-s2, -s1, s2], dtype=np.int32))
            gamma += path_forward + path_backward
    return gamma


@njit
def compute_gamma_improved(links, x, mu, u0):
    """
    Computes the contribution of the neighbours of the link $U_{\\mu}(x)$ to the improved action (103), yet to be multiplied on the left by $U_{\\mu}(x)$ (see eq. (113)).

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        x(numpy.ndarray[int, d]): the spacetime coordinates array of the factorized link
        mu(int): the direction of the factorized link

    Returns:
        numpy.ndarray[complex, 3, 3]: the matrix $\\Gamma_{\\mu}(x)$
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    gamma = np.zeros((3, 3), dtype=np.complex128)
    # move to the end of the link U_mu(x)
    x[mu] += 1
    # implement periodic boundary conditions
    if x[mu] >= N:
        x[mu] -= N
    # there are 2 plaquettes and 6 rectangles for each direction
    for nu in range(d):
        if nu != mu:
            # convert the spacetime index (from 0 to d-1) to steps (from 1 to d)
            s1 = mu + 1
            s2 = nu + 1

            # compute the plaquette contributions (2 plaquettes)
            plaquette_path_forward = compute_path(links, x, np.array([s2, -s1, -s2], dtype=np.int32))
            plaquette_path_backward = compute_path(links, x, np.array([-s2, -s1, s2], dtype=np.int32))
            plaquette_contributions = plaquette_path_forward + plaquette_path_backward

            # compute the rectangle contributions
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

            # add the contributions of the plaquette and the rectangles to Gamma_mu(x)
            gamma += 5 / 3 * 1 / u0**4 * plaquette_contributions - 1 / u0**6 * 1 / 12 * rectangle_contributions
    return gamma


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
        # update U -> MU
        M = pick_random_matrix(random_matrices)
        links[i][mu] = M @ links[i][mu]
        # compute the action change after the update
        dS = -beta / 3 * np.real(np.trace((links[i][mu] - old_U) @ gamma))
        # check Metropolis acceptance condition, and if it fails restore the previous link
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            links[i][mu] = old_U


@njit
def update_lattice(links, hits, beta, random_matrices, u0, improved):
    """
    Perform a sweep through the lattice of link updates.

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

    # for each spacetime point and for each direction, there is a U matrix to update
    for i in range(N**d):
        for mu in range(d):
            update_link(links, decode_index(i, N, d), mu, hits, beta, random_matrices, u0, improved)


@njit
def generate_wilson_samples(links, loops, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved, rotate_time=True):
    """
    Performs the metropolis algorithm to generate samples that will contribute to the path integral average.

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        loops(numpy.ndarray[int, n_loops, max_loop_length]): the list of loops to compute
        N_cf(int): the total number of samples to be generated
        N_cor(int): the number of path updates before picking each sample
        hits(int): the number of updates of each link before going to the next
        thermalization_its(int): the number of times that N_cor samples are discarded at the beginning to let the procedure thermalize
        bin_size(int): the number of samples to be averaged in a single bin
        beta(float): the beta parameter that enters in the calculation of the Wilson action
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices from which the update matrices are drawn
        u0(float): the u0 parameter that enters in the calculation of the Wilson action
        improved(bool): whether to use the improved action or not
        rotate_time(bool): whether or not to rotate time dimension when exploiting rotational symmetry of the lattice (if spatial directions have been smeared, set this to False)

    Returns:
        numpy.ndarray[float, N_bins, steps.shape[0]]: the matrix of Wilson loop samples
    """
    # detect the spacetime dimensions d and the number of lattice points N from the shape of the links data structure
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    n_loops_to_compute = loops.shape[0]
    N_bins = int(np.ceil(N_cf / bin_size))

    # prepare the arrays of samples. There is an additional index because there are more loops to compute and save.
    wilson_samples = np.zeros((N_bins, n_loops_to_compute), dtype=np.float64)
    bin_samples = np.zeros((bin_size, n_loops_to_compute), dtype=np.float64)

    # thermalization updates
    for i in range(thermalization_its):
        print(f"{i}/{thermalization_its} thermalization iteration")
        for _ in range(N_cor):
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
        print(f"{i}/{N_cf}")

        # discard N_cor values to thermalize
        for _ in range(N_cor):
            update_lattice(links, hits, beta, random_matrices, u0, improved)

        # there are more loops to compute
        for j in range(n_loops_to_compute):
            # sweep through all possible loops of the current kind in the lattice, also exploiting rotational symmetry
            value = 0  # this will contain the sum of all the contributions for each rotational configurations
            loop = loops[j, :]
            # for each rotational configuration, sweep through all spacetime points and compute the loop
            for l in range(rot_factor):
                # TODO: factorize
                partial_value = 0  # this will contain the value for the current rotational configuration, after sweeping through all spacetime points
                directions = directions_permutations[l]
                # rotate the steps in the loop onto the new axes
                rotated_loop = np.zeros_like(loop)
                for m in range(loop.shape[0]):
                    rotated_loop[m] = np.sign(loop[m]) * (directions[np.abs(loop[m]) - 1] + 1)
                # sweep through all spacetime points and add up the loop values
                for k in range(N**d):
                    y = decode_index(k, N, d)
                    path = compute_path(links, y, rotated_loop)
                    partial_value += 1 / 3 * np.real(np.trace(path))
                # average the loops value and add them to the total
                value += partial_value / (N**d)
            # average the contributions given by each rotational configuration and save them in the bin
            value /= rot_factor
            bin_samples[i % bin_size][j] = value

        # average the samples in the current bin and save the result in the samples array
        if (i + 1) % bin_size == 0 or i == N_cf - 1:
            for j in range(n_loops_to_compute):
                wilson_samples[i // bin_size][j] = bin_samples[:, j].mean()

    return wilson_samples


@njit(parallel=True)
def compute_path_integral_average(
    links, loops, N_cf, N_cor, hits, thermalization_its, N_copies, bin_size, beta, random_matrices, u0, improved, rotate_time=True
):
    """
    Uses the metropolis algorithm to calculate Wilson loop averages and errors

    Args:
        links(numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        loops(numpy.ndarray[int, n, d]): the list of loops to compute
        N_cf(int): the total number of samples to be generated
        N_cor(int): the number of path updates before picking each sample.
        hits(int): the number of updates.
        thermalization_its(int): the number of samples to be discarded at the beginning to let the procedure thermalize.
        N_copies(int): the number of copies for the bootstrap procedure.
        bin_size(int): the number of samples to be averaged in a single bin.
        beta(float): the beta parameter that enters in the calculation of the Wilson action.
        random_matrices(numpy.ndarray[complex, N * 2, 3, 3]): the set of random matrices.
        u0(float): the u0 parameter that enters in the calculation of the Wilson action.
        improved(bool): whether to use the improved Gamma matrix or not.
        rotate_time(bool): whether or not to rotate time dimension when exploiting rotational symmetry of the lattice (if spatial directions have been smeared, set this to False).
    Returns:
        np.ndarray[float, N_bins, steps.shape[0]]: the matrix of Wilson loop samples
    """
    # detect the spacetime dimensions d and the number of lattice points N from the shape of the links data structure
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    N_bins = int(np.ceil(N_cf / bin_size))  # if bin_size == 1, then N_bins == N_cf
    n_loops_to_compute = loops.shape[0]

    # generate an array of samples
    wilson_samples = generate_wilson_samples(
        links, loops, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved, rotate_time
    )

    # bootstrap procedure
    if N_copies > 1:
        # prepare array for output
        bootstrap_avgs = np.zeros((N_copies, n_loops_to_compute), dtype=np.float64)
        # we want to generate N_copies bootstraps
        for i in range(N_copies):
            # prepare array for bootstrap copy
            values = np.zeros((N_bins, n_loops_to_compute), dtype=np.float64)
            # draw N_bins random samples from wilson_samples and save them into 'values'
            for j in range(N_bins):
                index_of_copied_value = int(np.random.uniform(0, N_bins))
                for k in range(n_loops_to_compute):
                    values[j, k] = wilson_samples[index_of_copied_value, k]
            # average and save the result
            for k in range(n_loops_to_compute):
                bootstrap_avgs[i, k] = values[:, k].mean()
        return bootstrap_avgs
    else:
        # if N_copies is 1, no bootstrap
        return wilson_samples


@njit
def get_steps_for_rectangle(width, height, mu, nu):
    """
    Returns the steps for a rectangle, in step notation.

    Args:
        width(int): the length of the rectangle in mu direction
        height(int): the length of the rectangle in nu direction
        mu(int): the first direction
        nu(int): the second direction

    Returns:
        numpy.ndarray[int, 2 * (width + height)]: the steps for the rectangle.
    """
    # convert the steps in step notation (from 1 to d-1)
    s1 = mu + 1
    s2 = nu + 1
    # compute perimeter of the rectangle
    length = 2 * (width + height)
    # compute steps needed to draw the rectangle
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


@njit
def get_nonplanar_steps(widths):
    # TODO: implement better
    """


    Args:
        widths (int): the sides of the spacial parallelepiped that separates the two quarks

    Returns:
        numpy.ndarray[int]: a list of loops that goes from
    """
    # e.g. if widths == [2,3,3] then we need [+1,+1,+2,+2,+2,+3,+3,+3,-1,-1,-2,-2,-2,-3,-3,-3]
    tot_steps = np.sum(widths) * 2
    steps = np.zeros(tot_steps, dtype=np.int32)
    j = 0
    for i in range(widths.shape[0]):
        s = i + 1
        for _ in range(widths[i]):
            steps[j] = s
            steps[j + tot_steps // 2] = -s
            j += 1
    return steps


# computes a^2 \Delta^2 U
@njit
def compute_gauge_covariant_derivative(links, x, mu, u0):
    """
    Computes the gauge covariant derivative of a given link as in eq. (123)

    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        x (numpy.ndarray[int, d]): the spacetime coordinates of the point associated to the link in question
        mu (int): the direction.
        u0 (float): the u0 parameter that enters in the calculation of the Wilson action.

    Returns:
        numpy.ndarray[complex, 3, 3]: the gauge covariant derivative
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    # we need to sum over all directions
    D = np.zeros((3, 3), dtype=np.complex128)
    for rho in range(d):
        # convert spacetime directions into step notation
        s1 = mu + 1
        s2 = rho + 1
        # compute the three terms contributing to the covariant derivative
        path1 = compute_path(links, x, np.array([s2, s1, -s2]))
        path2 = compute_path(links, x, np.array([-s2, s1, s2]))
        path3 = -2 * u0**2 * get_link(links, x, s1)
        # add to the total
        D += path1 + path2 + path3
    # divide by u0^2 but not by a^2 because it gets simplified in the smearing operator
    return D / u0**2


@njit
def project_to_SU3(U):
    """
    Projects a matrix onto the group SU(3)

    Args:
        U (np.ndarray[complex, 3, 3]): the matrix to project onto SU(3)

    Returns:
        np.ndarray[complex, 3, 3]: the projection of U onto SU(3)
    """
    # make U unitary using Gram-Schmidt
    # orthogonalize the first and second row using Gram-Schmidt
    U[0] = U[0] / np.sqrt(np.sum(np.abs(U[0]) ** 2))
    U[1] = U[1] - np.dot(U[0].conj(), U[1]) * U[0]
    U[1] = U[1] / np.sqrt(np.sum(np.abs(U[1]) ** 2))

    # the third row is determined by unitarity, it must be orthogonal to the first two rows
    U[2] = np.cross(U[0].conj(), U[1].conj()).conj()

    # ensure determinant is 1
    det_U = np.linalg.det(U)
    U = U / (det_U ** (1 / 3))

    return U


@njit
def smear_matrix(old_links, new_links, x, mu, u0, eps):
    """
    Applies the smearing operator of eq. (121) once to a given link

    Args:
        old_links (np.ndarray[complex, N^d, d, 3, 3]): the links data structure containing the links before smearing
        new_links (np.ndarray[complex, N^d, d, 3, 3]): the links data structure where the smeared links have to be stored
        x (np.ndarray[int, d]): the spacetime point associated to the link to be smeared
        mu (int): the spacetime direction of the link to be smeared
        u0 (float): the u0 parameter that enters in the calculation of the Wilson action
        eps (float): the epsilon parameter for the smearing operator
    """
    d = old_links.shape[1]
    N = np.int32(old_links.shape[0] ** (1 / d))

    # get the original link from the old links (not yet smeared)
    i = encode_index(x, N, d)
    original_link = get_link(old_links, x, mu + 1)  # mu+1 is step notation

    # compute the gauge covariant derivative based on old links
    D = compute_gauge_covariant_derivative(old_links, x, mu, u0)

    # apply the smearing operator
    new_link = original_link + eps * D

    # update the new links array
    new_links[i][mu] = new_link


@njit
def smear_links(links, mus, u0, eps, n):
    """
    Smears all the links in the given spacetime directions

    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        mus (int): the spacetime directions to smear
        u0 (float): the u0 parameter that enters in the calculation of the Wilson action
        eps (float): the epsilon parameter of the smearing operator
        n (int): the number of smearing updates to apply

    Returns:
        np.ndarray[complex, N^d, d, 3, 3]: the links data structure containing the smeared links

    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    # copy links to a read-only data structure, to be used for derivative calculation
    old_links = np.copy(links)
    # prepare the data structure for the smeared links
    new_links = np.zeros_like(links)

    for _ in range(n):
        # for each node, get the link and apply the smearing operator to all the specified directions
        for mu in mus:
            for i in range(links.shape[0]):
                x = decode_index(i, N, d)
                smear_matrix(old_links, new_links, x, mu, u0, eps)
        # after all the links have been smeared, update the data structure containing the links to be used for computing derivatives in the next iteration
        old_links = np.copy(new_links)
    return new_links


@njit(parallel=True)
def compute_static_quark_potential(
    N,
    d,
    N_cf,
    N_cor,
    hits,
    thermalization_its,
    N_copies,
    bin_size,
    beta,
    random_matrices,
    u0,
    improved,
    width_t,
    max_r,
    eps_smearing=0.0,
    n_smearing=0,
):

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
    possible_separations = get_non_decreasing_sequences(d - 1, N - 1)
    accepted_separations = np.zeros_like(possible_separations)
    num_separations = 0
    for i in range(possible_separations.shape[0]):
        if 0 < np.sum(possible_separations[i] ** 2) <= max_r**2:
            accepted_separations[num_separations] = possible_separations[i]
            num_separations += 1

    # for each spatial separation, there are two loops to compute, one with t and the other with (t+a) temporal separation
    loops = np.zeros((num_separations * 2, max_length_loop), dtype=np.int32)

    # prepare the coordinates' deltas for the spatial separation, fixing the time separations
    x_t = np.zeros(d, dtype=np.int32)
    x_t_a = np.zeros(d, dtype=np.int32)
    x_t[0] = width_t
    x_t_a[0] = width_t_a

    # for each spatial separation, write down the steps for the two loops
    for i in range(num_separations):
        x = accepted_separations[i]
        # fill x_t and x_t_a in the spatial indices
        for j in range(d - 1):
            x_t[j + 1] = x[j]
            x_t_a[j + 1] = x[j]
        # compute the steps sequence to reach the two points in spacetime
        steps_t = get_nonplanar_steps(x_t)
        steps_t_a = get_nonplanar_steps(x_t_a)

        # insert the loops in the full loop list
        for j in range(steps_t.shape[0]):
            loops[i, j] = steps_t[j]
        for j in range(steps_t_a.shape[0]):
            loops[i + num_separations, j] = steps_t_a[j]

    # compute path integral averages for all loops
    results = compute_path_integral_average(
        links,
        loops,
        N_cf,
        N_cor,
        hits,
        0,
        N_copies,
        bin_size,
        beta,
        random_matrices,
        u0,
        improved,
        rotate_time=False,  # the lattice is already thermalized
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
