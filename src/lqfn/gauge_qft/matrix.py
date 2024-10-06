import numpy as np
from numba import njit
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
