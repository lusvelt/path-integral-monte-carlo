import numpy as np
from numba import njit


@njit
def factorial(n):
    f = 1
    for i in range(1, n + 1):
        f *= i
    return f


@njit
def generate_sequences(result, current_sequence, pos, max_digit, idx):
    if pos == len(current_sequence):
        result[idx[0], :] = current_sequence.copy()  # Store the current sequence
        idx[0] += 1
        return

    start = current_sequence[pos - 1] if pos > 0 else 0  # Ensure non-decreasing order
    for digit in range(start, max_digit + 1):
        current_sequence[pos] = digit
        generate_sequences(result, current_sequence, pos + 1, max_digit, idx)


@njit
def get_non_decreasing_sequences(length, max_digit):
    num_sequences = factorial(max_digit + length) // (factorial(length) * factorial(max_digit))  # Calculate number of sequences
    result = np.zeros((num_sequences, length), dtype=np.int32)  # Store the sequences
    current_sequence = np.zeros(length, dtype=np.int32)
    idx = np.array([0])  # Index to keep track of the next position in result

    generate_sequences(result, current_sequence, 0, max_digit, idx)
    return result


@njit
def generate_permutations(arr, start, end, result, idx):
    if start == end:
        result[idx[0], :] = arr.copy()  # Store the permutation
        idx[0] += 1
    else:
        for i in range(start, end):
            arr[start], arr[i] = arr[i], arr[start]  # Swap
            generate_permutations(arr, start + 1, end, result, idx)
            arr[start], arr[i] = arr[i], arr[start]  # Backtrack


@njit
def permute(arr):
    N = arr.shape[0]
    num_permutations = factorial(N)  # Total number of permutations
    result = np.zeros((num_permutations, N), dtype=np.int32)  # Store permutations
    idx = np.array([0])  # Track the index in result
    generate_permutations(arr, 0, N, result, idx)
    return result
