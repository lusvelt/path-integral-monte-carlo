from lqfn.numerical.pimc import metropolis
import numpy as np

N = 10
a = 1


# Consider a non-relativistic particle with a harmonic potential
def S_per_timeslice(j, x):
    jp = (j + 1) % N  # next site
    jm = (j - 1) % N  # previous site
    return a * x[j] ** 2 / 2 + x[j] * (x[j] - x[jp] - x[jm]) / a


def test_compute_path_integral_average():
    precision = 0.9
    N_tests = 100
    err = 4

    # We want to average the functional \Gamma[x] = x(t)
    # We expect the result to be x(t) = 0
    def functional(x, n):
        return x[n]

    # Perform N_tests tests and assert that the expectation value is close to 0 within err standard deviations in at least
    # a fraction of N_tests equal to precision
    passed = 0
    for _ in range(N_tests):
        avg, std = metropolis.compute_path_integral_average(functional, S_per_timeslice, N, N_cf=200, N_cor=5, eps=1.5)
        min = avg - err * std
        max = avg + err * std
        if (min < np.zeros(N)).all() and (np.zeros(N) < max).all():
            passed += 1
    print(f"{passed}/{N_tests}, while desired: {int(precision*N_tests)}/{N_tests}")
    assert passed >= precision * N_tests
