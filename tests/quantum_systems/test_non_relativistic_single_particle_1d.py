import pytest
import math
import numpy as np
from scipy import special as sc
from lqfn.quantum_systems import NonRelativisticSingleParticle1D


m = 1


def V(x):
    return m * x**2 / 2


T = 5
N = 10
a = 0.5
system = NonRelativisticSingleParticle1D(V, T=5, m=m, N=10, box=(-10, 10))


def test_NonRelativisticSingleParticle1D():
    assert isinstance(system, NonRelativisticSingleParticle1D)
    assert isinstance(system.box, np.ndarray)
    assert system.a == a


def test_S_lat():
    # Consider a path of all zeros: the action should be zero
    path = np.zeros(N + 1)
    assert np.isclose(system.S_lat(path), 0)
    # Now perturb the path and check the action is different from zero
    path[0] = 0.1
    S = system.S_lat(path)
    assert S > 0
    assert not np.isclose(S, 0)


def known_energy(n: int):
    return 1 / 2 + n


def known_eigenstate(n: int, x):
    return 1 / np.sqrt(2**n * math.factorial(n)) * np.pi ** (-1 / 4) * np.exp(-(x**2) / 2) * sc.hermite(n)(x)


def test_compute_propagator():
    # Compute propagator with MC and analytically (in the limit T -> inf)
    # and check that the results are compatible within a large percentage of error (its a fast test so precision is low)
    x = np.linspace(-2.0, 2.0, 20)
    propagators_exact = system.compute_propagator_from_ground_state(x)
    propagators_pimc_results = system.compute_propagator_pimc(x, nitn_tot=30, nitn_discarded=10, neval=1000, lower_bound=-5, upper_bound=5)
    propagators_pimc = np.array([p.mean for p in propagators_pimc_results])
    # Check that the integration procedure gives meaningful results in at least percentage fraction of the points
    percentage = 0.75

    ok = 0
    for result in propagators_pimc_results:
        if result.Q >= 0.01:
            ok += 1
    assert ok >= int(len(propagators_pimc_results) * percentage)

    ok = 0
    for i in range(len(propagators_pimc)):
        if np.isclose(propagators_exact[i], propagators_pimc[i], rtol=0.25):
            ok += 1
    assert ok >= int(len(propagators_pimc_results) * percentage)


def test_solve_schrodinger():
    # Compare the solution given by the implemented function with the known one
    N_schr = 100
    computed_eigenstates = system.solve_schrodinger(N=N_schr, max_states=3)
    for n in range(len(computed_eigenstates.energies)):
        assert np.isclose(computed_eigenstates.energies[n], known_energy(n), rtol=0.01)
        x = np.linspace(system.box[0], system.box[1], N_schr)
        delta1 = computed_eigenstates.array[n] - known_eigenstate(n, x)
        delta2 = -computed_eigenstates.array[n] - known_eigenstate(n, x)
        norm1 = np.linalg.norm(delta1, 2)
        norm2 = np.linalg.norm(delta2, 2)
        # We don't know the relative phase, so we check both
        assert np.isclose(norm1, 0, atol=0.05) or np.isclose(norm2, 0, atol=0.05)
