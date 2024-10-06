"""
This module contains numerical recipes to perform basic tasks, such as root finding.
"""

from typing import Callable
import concurrent.futures
import numpy as np


def select_root_subinterval(f: Callable, a: float, b: float, num_subintervals: int) -> tuple[float, float]:
    """
    Divides the interval $[a,b]$ into `(num_cores - 1)` subintervals and returns the first subinterval $[c, d]$ where $f(c)f(d) < 0$.
    The function $f$ and the interval $[a,b]$ must be such that $f(a)f(b) < 0$
    To compute the value of the function on the edges, multithreading is used, since f can be computationally heavy to evaluate.

    Args:
        f (function[float]->float): A regular real function
        a (float): The left extreme of the interval
        b (float): The right extreme of the interval
        num_subintervals (int): The number of subintervals to divide $[a,b]$ into

    Returns
        float: The left edge of the found subinterval
        float: The right edge of the found subinterval
    """
    num_extremes = num_subintervals + 1
    # Divide the interval where we expect the root to be into num_cores-1 subintervals
    extremes = np.linspace(a, b, num_extremes)
    # Compute the values of the function at each extreme
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_extremes) as executor:
        futures = [executor.submit(f, z) for z in extremes]
        results = [job.result() for job in futures]
    i_first = 0
    i_last = num_subintervals - 1
    if results[0] * results[-1] < 0:
        # Select the subinterval whose extremes have opposite signs
        for i in range(num_subintervals):
            if results[i] * results[i + 1] < 0:
                i_first = i
                break
        for i in range(num_subintervals - 1, 0, -1):
            if results[i] * results[i - 1] < 0:
                i_last = i
                break
        assert i_first < i_last
    return extremes[i_first], extremes[i_last]


def find_root(f: callable, a: float, b: float, num_cores: int, max_precision=1e-3) -> float:
    """
    Finds the root within the interval $[a,b]$ of a function $f$ via recursive (generalized) bisection.
    Multithreading is used to evaluate the function at the various points.

    Args:
        f (function[float]->float): A regular real function
        a (float): The left extreme of the interval
        b (float): The right extreme of the interval
        num_cores (int): The number of CPU cores to use for the process
        abs_precision (float): The absolute precision desired for the root

    Returns:
        float: The estimated root
        err: The error associated to the root
    """
    assert b > a
    assert 1 <= num_cores <= 32
    assert max_precision > 0
    new_a, new_b = select_root_subinterval(f, a, b, num_cores - 1)
    print(f"Narrowed down to [{new_a}, {new_b}]")
    # If precision is reached, or the results start to oscillate, return the root
    if new_b - new_a < max_precision or (new_b - new_a) > (b - a) / (num_cores - 1) * 1.5:
        return (new_b + new_a) / 2, (new_b - new_a) / 2
    else:
        return find_root(f, new_a, new_b, num_cores, max_precision)
