"""
This module contains functions to compute quantities in lattice QCD.
"""

import numpy as np
from numba import njit


@njit
def factorial(n):
    f = 1
    for i in range(1, n + 1):
        f *= i
    return f


@njit
def generate_random_SU3_update_matrix(eps, taylor_order=20):
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
    result = np.zeros(d, dtype=np.int32)
    result = 0
    for i in range(d):
        result += N**i * x[i]
    return result


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
    assert x.shape == (d,)
    assert 0 <= mu < d
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
    assert x.shape == (d,)
    assert 0 <= mu < d
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
            gamma += 5 / (3 * u0**4) * plaquette_contributions - 1 / u0**6 * 1 / 12 * rectangle_contributions
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
    assert x.shape == (d,)
    assert 0 <= mu < d
    assert hits > 0
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
def generate_wilson_samples(links, x, steps, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved):
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    N_bins = int(np.ceil(N_cf / bin_size))
    wilson_samples = np.zeros(N_bins, dtype=np.float64)
    bin_samples = np.zeros(bin_size, dtype=np.float64)
    for _ in range(thermalization_its * N_cor):  # thermalization
        update_lattice(links, hits, beta, random_matrices, u0, improved)
    for i in range(N_cf):
        print(f"{i}/{N_cf}")
        for _ in range(N_cor):  # discard N_cor values
            update_lattice(links, hits, beta, random_matrices, u0, improved)
        path = compute_path(links, x, steps)
        bin_samples[i % bin_size] = 1 / 3 * np.real(np.trace(path))
        if (i + 1) % bin_size == 0 or i == N_cf - 1:
            wilson_samples[i // bin_size] = bin_samples.mean()
    return wilson_samples


@njit
def compute_path_integral_average(links, x, steps, N_cf, N_cor, hits, thermalization_its, N_copies, bin_size, beta, random_matrices, u0, improved):
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))

    assert N_copies <= N_cf
    assert bin_size <= N_cf

    wilson_samples = generate_wilson_samples(links, x, steps, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved)
    N_bins = int(np.ceil(N_cf / bin_size))  # if bin_size == 1, then N_bins == N_cf
    if N_copies > 1:  # bootstrap procedure
        bootstrap_avgs = np.zeros(N_copies, dtype=np.float64)
        for i in range(N_copies):
            values = np.zeros(N_bins, dtype=np.float64)
            for j in range(N_bins):
                index_of_copied_value = int(np.random.uniform(0, N_bins))
                values[j] = wilson_samples[index_of_copied_value]
            bootstrap_avgs[i] = values.mean()
        return np.array([bootstrap_avgs.mean(), bootstrap_avgs.std()], dtype=np.float64)
    else:
        avg = wilson_samples.mean()
        std = wilson_samples.std() / np.sqrt(N_bins)
        return np.array([avg, std], dtype=np.float64)
