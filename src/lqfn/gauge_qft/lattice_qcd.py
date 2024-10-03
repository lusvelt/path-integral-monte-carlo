"""
This module contains functions to compute quantities in lattice QCD.
"""

import numpy as np
from numba import njit, prange


@njit
def factorial(n):
    f = 1
    for i in range(1, n + 1):
        f *= i
    return f


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
def generate_wilson_samples(links, steps, N_cf, N_cor, hits, thermalization_its, bin_size, beta, random_matrices, u0, improved):
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
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    i = encode_index(x, N, d)
    links[i][mu] = np.identity(3) + eps * compute_gauge_covariant_derivative(links, x, mu, u0)
    if n > 1:
        smear_matrix(links, x, mu, u0, eps, n - 1)


@njit
def smear_links(links, mu, u0, eps, n):
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    old_links = np.copy(links)
    new_links = np.zeros_like(links)
    for i in range(links.shape[0]):
        x = decode_index(i, N, d)
        links = np.copy(old_links)
        smear_matrix(links, x, mu, u0, eps, n)
        new_links[i, mu] = np.copy(links[i, mu])


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
