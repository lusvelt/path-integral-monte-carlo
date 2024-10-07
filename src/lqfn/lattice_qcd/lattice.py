"""
This module contains utilities for manipulating a QCD lattice.
All the functions are written with numba, since these utilities will be called by functions which implement computationally intense procedures.
The main data structure that is carried around the various functions is called `links`, which contains the field configuration in terms of links connecting adjacent lattice points.
The shape of the links array is $(N^d, d, 3, 3)$:
- The first index is for the spacetime point $x$, encoded in a decimal integer
- The second index is for the spacetime direction
- The other two indices are for the SU(3) matrix
This data structure encodes all information about the lattice, in particular:
- `N`: the number of lattice points in each direction
- `d`: the spacetime dimensions
To avoid passing around `N` and `d` alongside `links` in each function, at the beginning of most functions we invoke the `get_d_N(links)` function to get this info.
"""

import numpy as np
from numba import njit
from ..combinatorics import *


@njit
def create_lattice_links(N, d):
    """
    Creates the lattice links encoding the field configuration.
    Each link is initialized to be the identity.

    Args:
        N (int): the number of lattice sites in each direction.
        d (int): the number of dimensions of the lattice.

    Returns:
        numpy.ndarray[complex, N^d, d, 3, 3]: the lattice links data structure
    """
    shape = (N**d, d, 3, 3)
    links = np.zeros(shape, dtype=np.complex128)
    for i in range(N**d):
        for mu in range(d):
            links[i][mu] = np.identity(3, dtype=np.complex128)
    return links


@njit
def get_d_N(links):
    """
    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): The lattice links data structure

    Returns:
        int: the number of spacetime dimensions
        int: the number of lattice points for each spacetime direction
    """
    d = links.shape[1]
    N = np.int32(links.shape[0] ** (1 / d))
    return d, N


@njit
def get_node(links, x):
    """
    Returns the node at a given multi-index.

    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): The lattice links data structure
        x (numpy.ndarray[int, d]): The spacetime coordinates of the node

    Returns:
        numpy.ndarray[complex, d, 3, 3]: a numpy array containing d links, one for each spacetime direction
    """
    d, N = get_d_N(links)
    i = encode_index(x, N, d)
    return links[i]


@njit
def pbc(x, d, N):
    """
    Ensures that periodic boundary conditions are satisfied, modifying the array if needed

    Args:
        x (np.ndarray[int, d]): the spacetime coordinates of the lattice point
    """
    for mu in range(d):
        while x[mu] < 0:
            x[mu] += N
        while x[mu] >= N:
            x[mu] -= N


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
    d, N = get_d_N(links)

    # The spacetime index is from 0 to d-1, while the step is from 1 to d with sign
    mu = np.abs(step) - 1
    x = np.copy(x)
    if step > 0:  # the step is in positive direction
        node = get_node(links, x)
        link = node[mu]
    else:  # the step in in negative direction
        # go to the previous node in that direction
        x[mu] -= 1
        pbc(x, d, N)
        node = get_node(links, x)
        # take the hermitian conjugate
        link = node[mu].conj().T
    return link


@njit
def get_spatial_separations(N, d, max_r):
    """
    Returns the list of vectors in the lattice that have norm <= max_r and the components in non-decreasing order

    Args:
        N (int): the number of lattice points in each spacetime direction
        d (int): the number of spacetime dimensions
        max_r (float): the cutoff value (inclusive) for the norm of accepted spatial separation vectors

    Returns:
        np.ndarray[int, num_separations, d]: the list of accepted spatial separations
    """
    # get all vectors in the lattice with components in non-decreasing order
    possible_separations = get_non_decreasing_sequences(d - 1, N - 1)
    # prepare array for accepted separations
    accepted_separations = np.zeros_like(possible_separations)
    num_separations = 0
    for i in range(possible_separations.shape[0]):
        # check that the norm is positive and less than max_r
        if 0 < np.sum(possible_separations[i] ** 2) <= max_r**2:
            accepted_separations[num_separations] = possible_separations[i]
            num_separations += 1
    # transfer the results in a smaller array, to be returned
    separations = np.zeros((num_separations, d), dtype=np.int32)
    for i in range(num_separations):
        separations[i] = accepted_separations[i]
    return separations


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
    d, N = get_d_N(links)
    # initialize variable that will contain the matrices products of the links
    Us = np.identity(3, dtype=np.complex128)
    x = np.copy(x_start)

    for step in steps:
        if step != 0:
            # perform the update
            Us = Us @ get_link(links, x, step)
            # the spacetime direction is from 0 to d-1
            mu = np.abs(step) - 1
            # move to the next node
            x[mu] += np.sign(step)
            pbc(x, d, N)
    return Us


@njit
def compute_partial_adjacent_plaquettes(links, x, mu, nu):
    """
    Computes the two paths in the nu direction that, after left multiplication with $U_{\\mu}(x)$ become plaquettes

    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        x (numpy.ndarray[int, d]): the spacetime coordinates array of the factorized link
        mu (int): the direction of the factorized link
        nu (int): the direction of the plaquette

    Returns:
        np.ndarray[complex, 3, 3]: the sum of the two partial plaquettes
    """
    # for each direction there are two plaquettes, except for the direction of the link itself

    # convert the spacetime index (from 0 to d-1) to steps (from 1 to d)
    s1 = mu + 1
    s2 = nu + 1
    # compute the remaining path, starting from the end of the link to its start, tracing three sides of the plaquette
    path_forward = compute_path(links, x, np.array([s2, -s1, -s2], dtype=np.int32))
    path_backward = compute_path(links, x, np.array([-s2, -s1, s2], dtype=np.int32))
    return path_forward + path_backward


@njit
def compute_partial_adjacent_rectangles(links, x, mu, nu):
    """
    Computes the six paths in the nu direction that, after left multiplication with $U_{\\mu}(x)$ become 2 by 1 rectangles

    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        x (numpy.ndarray[int, d]): the spacetime coordinates array of the factorized link
        mu (int): the direction of the factorized link
        nu (int): the direction of the plaquette

    Returns:
        np.ndarray[complex, 3, 3]: the sum of the six partial rectangles
    """
    # convert the spacetime index (from 0 to d-1) to steps (from 1 to d)
    s1 = mu + 1
    s2 = nu + 1

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
    return rectangle_contributions


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
    d, N = get_d_N(links)
    # move to the end of the link U_mu(x)
    x[mu] += 1
    pbc(x, d, N)
    gamma = np.zeros((3, 3), dtype=np.complex128)
    for nu in range(d):
        if nu != mu:
            gamma += compute_partial_adjacent_plaquettes(links, x, mu, nu)
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
    d, N = get_d_N(links)

    gamma = np.zeros((3, 3), dtype=np.complex128)
    # move to the end of the link U_mu(x)
    x[mu] += 1
    pbc(x, d, N)
    # there are 2 plaquettes and 6 rectangles for each direction
    for nu in range(d):
        if nu != mu:
            plaquette_contributions = compute_partial_adjacent_plaquettes(links, x, mu, nu)
            rectangle_contributions = compute_partial_adjacent_rectangles(links, x, mu, nu)

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
def rotate_loop(loop, rotated_axes):
    """
    Gets the loop in the rotated frame

    Args:
        loop (np.ndarray[int]): the loop to rotate
        rotated_axes (np.ndarray[int]): a permutation of the first d integers, representing the rotated frame.
            If the element of index `mu` has value `nu`, it means that the old axis `mu` has been rotated into the new axis `nu`.

    Returns:
        np.ndarray[int]: the loop after frame rotation
    """
    directions = rotated_axes
    # rotate the steps in the loop onto the new axes
    rotated_loop = np.zeros_like(loop)
    for m in range(loop.shape[0]):
        rotated_loop[m] = np.sign(loop[m]) * (directions[np.abs(loop[m]) - 1] + 1)
    return rotated_loop


@njit
def get_nonplanar_steps(widths):
    """
    Args:
        widths (int): the sides of the spacial parallelepiped that separates the two quarks

    Returns:
        numpy.ndarray[int]: a list of loops that goes from
    """
    # TODO: maybe there are more paths
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
    Computes the gauge covariant derivative of a given link as in eq. (123), multiplied by $a^2$.

    Args:
        links (numpy.ndarray[complex, N^d, d, 3, 3]): the lattice links data structure
        x (numpy.ndarray[int, d]): the spacetime coordinates of the point associated to the link in question
        mu (int): the direction.
        u0 (float): the u0 parameter that enters in the calculation of the Wilson action.

    Returns:
        numpy.ndarray[complex, 3, 3]: the gauge covariant derivative
    """
    d, N = get_d_N(links)

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
def smear_link(old_links, new_links, x, mu, u0, eps):
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
    d, N = get_d_N(old_links)

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
    d, N = get_d_N(links)

    # copy links to a read-only data structure, to be used for derivative calculation
    old_links = np.copy(links)
    # prepare the data structure for the smeared links
    new_links = np.zeros_like(links)

    # we need to perform n smearing sweeps
    for _ in range(n):
        # for each node, get the link and apply the smearing operator to all the specified directions
        for mu in mus:
            for i in range(links.shape[0]):
                x = decode_index(i, N, d)
                smear_link(old_links, new_links, x, mu, u0, eps)
        # after all the links have been smeared, update the data structure containing the links to be used for computing derivatives in the next iteration
        old_links = np.copy(new_links)
    return new_links
