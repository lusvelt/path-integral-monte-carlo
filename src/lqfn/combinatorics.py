"""
This module contains several combinatorics utility functions that are decorated with numba's @njit, since it doesn't allow for python itertools to be used.
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
def _generate_sequences(result, current_sequence, pos, max_digit, idx):
    if pos == len(current_sequence):
        result[idx[0], :] = current_sequence.copy()  # Store the current sequence
        idx[0] += 1
        return

    start = current_sequence[pos - 1] if pos > 0 else 0  # Ensure non-decreasing order
    for digit in range(start, max_digit + 1):
        current_sequence[pos] = digit
        _generate_sequences(result, current_sequence, pos + 1, max_digit, idx)


@njit
def get_non_decreasing_sequences(length, max_digit):
    num_sequences = factorial(max_digit + length) // (factorial(length) * factorial(max_digit))  # Calculate number of sequences
    result = np.zeros((num_sequences, length), dtype=np.int32)  # Store the sequences
    current_sequence = np.zeros(length, dtype=np.int32)
    idx = np.array([0])  # Index to keep track of the next position in result

    _generate_sequences(result, current_sequence, 0, max_digit, idx)
    return result


@njit
def _generate_permutations(arr, start, end, result, idx):
    if start == end:
        result[idx[0], :] = arr.copy()  # Store the permutation
        idx[0] += 1
    else:
        for i in range(start, end):
            if arr[start] != arr[i]:
                arr[start], arr[i] = arr[i], arr[start]  # Swap
                _generate_permutations(arr, start + 1, end, result, idx)
                arr[start], arr[i] = arr[i], arr[start]  # Backtrack


@njit
def permute(arr):
    N = arr.shape[0]
    max_num_permutations = factorial(N)  # Total number of permutations
    result = np.zeros((max_num_permutations, N), dtype=np.int32)  # Store permutations
    idx = np.array([0])  # Track the index in result
    _generate_permutations(arr, 0, N, result, idx)
    # now idx[0] contains the number of generated permutations
    actual_permutations = np.zeros((idx[0], N), dtype=np.int32)
    for i in range(idx[0]):
        actual_permutations[i] = result[i]
    return actual_permutations


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
