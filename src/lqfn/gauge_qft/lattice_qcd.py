"""
This module contains functions to compute quantities in lattice QCD.
"""

import numpy as np
from numba import njit
from .combinatorics import *


@njit
def generate_random_SU3_update_matrix(eps, taylor_order=50):
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
    assert 0 < eps < 1
    s = np.zeros((N * 2, 3, 3), dtype=np.complex128)
    for i in range(N):
        s[i] = generate_random_SU3_update_matrix(eps)
        s[N + i] = s[i].conj().T
    return s


@njit
def decode_index(index, N, d):
    result = np.zeros(d, dtype=np.int32)
    for i in range(d):
        result[i] = index % N
        index = index // N
    return result


@njit
def encode_index(x, N, d):
    result = 0
    for i in range(d):
        result += N**i * x[i]
    return np.int32(result)


@njit
def create_lattice_links(N, d):
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
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    # Returns a numpy array containing d matrices, one for each spacetime direction
    i = encode_index(x, N, d)
    return links[i]


@njit
def get_link(links, x, step):
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
    s1 = mu + 1
    s2 = nu + 1
    steps = [s1, s2, -s1, -s2]
    path = compute_path(links, x, steps)
    return 1 / 3 * np.real(np.trace(path))


@njit
def compute_wilson_action(links, beta):
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
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    assert x.shape == (d,)
    assert 0 <= mu < d
    U = get_link(links, x, mu + 1)
    return -beta / 3 * np.real(np.trace(U @ gamma))


@njit
def pick_random_matrix(random_matrices):
    i = np.random.choice(random_matrices.shape[0])
    return random_matrices[i]


@njit
def update_link(links, x, mu, hits, beta, random_matrices, u0, improved):
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
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    for i in range(N**d):
        for mu in range(d):
            update_link(links, decode_index(i, N, d), mu, hits, beta, random_matrices, u0, improved)


@njit
def generate_wilson_samples(links, loops, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved, rotate_time=True):
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    n_loops_to_compute = loops.shape[0]

    N_bins = int(np.ceil(N_cf / bin_size))
    wilson_samples = np.zeros((N_bins, n_loops_to_compute), dtype=np.float64)
    bin_samples = np.zeros((bin_size, n_loops_to_compute), dtype=np.float64)

    for i in range(thermalization_its):  # thermalization
        print(f"{i}/{thermalization_its} thermalization iteration")
        for _ in range(N_cor):
            update_lattice(links, hits, beta, random_matrices, u0, improved)

    directions_permutations = permute(np.arange(d))
    rot_factor = factorial(d)
    if rotate_time is False:
        rot_factor = factorial(d - 1)

    for i in range(N_cf):

        print(f"{i}/{N_cf}")

        for _ in range(N_cor):  # discard N_cor values
            update_lattice(links, hits, beta, random_matrices, u0, improved)

        for j in range(n_loops_to_compute):
            # sweep through all possible loops of the current kind in the lattice
            value = 0
            loop = loops[j, :]
            for l in range(rot_factor):
                partial_value = 0
                directions = directions_permutations[l]
                # rotate the steps in the loop onto the new axes
                rotated_loop = np.zeros_like(loop)
                for m in range(loop.shape[0]):
                    rotated_loop[m] = np.sign(loop[m]) * (directions[np.abs(loop[m]) - 1] + 1)
                for k in range(N**d):
                    y = decode_index(k, N, d)
                    path = compute_path(links, y, rotated_loop)
                    partial_value += 1 / 3 * np.real(np.trace(path))
                value += partial_value / (N**d)
            value /= rot_factor
            bin_samples[i % bin_size][j] = value

        if (i + 1) % bin_size == 0 or i == N_cf - 1:
            for j in range(n_loops_to_compute):
                wilson_samples[i // bin_size][j] = bin_samples[:, j].mean()
                print(wilson_samples[i // bin_size][j])

    return wilson_samples


@njit(parallel=True)
def compute_path_integral_average(
    links, loops, N_cf, N_cor, hits, thermalization_its, N_copies, bin_size, beta, random_matrices, u0, improved, rotate_time=True
):
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    assert N_copies <= N_cf
    assert bin_size <= N_cf

    n_loops_to_compute = loops.shape[0]

    wilson_samples = generate_wilson_samples(
        links, loops, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved, rotate_time
    )
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


@njit
def get_nonplanar_steps(widths):
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
def gram_schmidt_unitary_projection(U):
    # Orthogonalize the first and second row using Gram-Schmidt
    U[0] = U[0] / np.sqrt(np.sum(np.abs(U[0]) ** 2))
    U[1] = U[1] - np.dot(U[0].conj(), U[1]) * U[0]
    U[1] = U[1] / np.sqrt(np.sum(np.abs(U[1]) ** 2))

    # The third row is determined by unitarity, it must be orthogonal to the first two rows
    U[2] = np.cross(U[0].conj(), U[1].conj()).conj()
    return U


@njit
def project_to_SU3(U):
    # Make U unitary using Gram-Schmidt
    U = gram_schmidt_unitary_projection(U)

    # Ensure determinant is 1 (SU(3))
    det_U = np.linalg.det(U)
    U = U / (det_U ** (1 / 3))

    return U


@njit
def smear_matrix(old_links, new_links, x, mu, u0, eps):
    d = old_links.shape[1]
    N = np.int32(old_links.shape[0] ** (1 / d))
    i = encode_index(x, N, d)

    # Get the original link from the old links (not yet smeared)
    original_link = get_link(old_links, x, mu + 1)

    # Compute the gauge covariant derivative based on old links
    D = compute_gauge_covariant_derivative(old_links, x, mu, u0)

    # Apply the smearing update: Add the contribution to the link
    new_link = original_link + eps * D

    # Project the matrix back to SU(3) to ensure unitarity
    new_link = project_to_SU3(new_link)

    # Update the new links array
    new_links[i][mu] = new_link


@njit
def smear_links(links, mu, u0, eps, n):
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    old_links = np.copy(links)
    new_links = np.copy(links)

    for _ in range(n):
        for i in range(links.shape[0]):
            x = decode_index(i, N, d)
            smear_matrix(old_links, new_links, x, mu, u0, eps)
        old_links = np.copy(new_links)
    return new_links


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
    links = create_lattice_links(N, d)

    # Smearing
    if eps_smearing != 0.0 and n_smearing != 0:
        # Smear all spatial directions
        for mu in range(1, d):
            smear_links(links, mu, u0, eps_smearing, n_smearing)

    # for each spatial separation we need two loops with t and t+a temporal separation
    width_t_a = width_t + 1

    # compute the length for the most lengthy loop
    max_length_loop = (N - 1) * d * 2

    # Prepare loops for all spatial separations of the two quarks
    x_t = np.zeros(d, dtype=np.int32)
    x_t_a = np.zeros(d, dtype=np.int32)
    x_t[0] = width_t
    x_t_a[0] = width_t_a

    possible_steps = get_non_decreasing_sequences(d - 1, N - 1)
    accepted_steps = np.zeros_like(possible_steps)
    num_loops = 0
    for i in range(possible_steps.shape[0]):
        if 0 < np.sum(possible_steps[i] ** 2) < max_r**2:
            accepted_steps[num_loops] = possible_steps[i]
            num_loops += 1

    loops = np.zeros((num_loops * 2, max_length_loop), dtype=np.int32)

    for i in range(num_loops):
        x = accepted_steps[i]
        for j in range(d - 1):
            x_t[j + 1] = x[j]
            x_t_a[j + 1] = x[j]
        steps_t = get_nonplanar_steps(x_t)
        steps_t_a = get_nonplanar_steps(x_t_a)

        for j in range(steps_t.shape[0]):
            loops[i, j] = steps_t[j]
        for j in range(steps_t_a.shape[0]):
            loops[i + num_loops, j] = steps_t_a[j]
        print(loops[i])
        print(loops[i + num_loops])

    # compute path integral averages for all loops
    results = compute_path_integral_average(
        links, loops, N_cf, N_cor, hits, thermalization_its, N_copies, bin_size, beta, random_matrices, u0, improved, rotate_time=False
    )

    # Bootstrap
    V_bootstrap = np.zeros((N_copies, num_loops), dtype=np.float64)
    for i in range(N_copies):
        for j in range(num_loops):
            V_bootstrap[i, j] = np.log(np.abs(results[i, j] / results[i, j + num_loops]))

    # [0]: r2, [1]: V, [2]: err
    return_data = np.zeros((3, num_loops), dtype=np.float64)
    for i in range(num_loops):
        x = accepted_steps[i]
        r = np.sqrt(np.sum(x**2))
        V = np.sum(V_bootstrap[:, i]) / N_copies
        err = np.sqrt(np.sum((V_bootstrap[:, i] - V) ** 2) / N_copies)
        return_data[0, i] = r
        return_data[1, i] = V
        return_data[2, i] = err

    return return_data
