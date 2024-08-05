import numpy as np
from scipy.optimize import brentq
# TODO: refine function signatures, and maybe set up a class

def numerov(E, dx, x, V):
    psi = np.zeros_like(x)
    psi[0] = 0
    psi[1] = 1e-5  # small initial value
    k2 = 2 * (E - V)
    for i in range(1, len(x) - 1):
        psi[i + 1] = (2 * psi[i] * (1 - 5 * dx**2 * k2[i] / 12) - psi[i - 1] * (1 + dx**2 * k2[i - 1] / 12)) / (1 + dx**2 * k2[i + 1] / 12)
    return psi


def find_eigenvalue(E, dx, x, V):
    psi = numerov(E, dx, x, V)
    return psi[-1]


def scan_energy_ranges(V, x, dx, energy_min, energy_max, num_points=100):
    energies = np.linspace(energy_min, energy_max, num_points)
    signs = np.array([np.sign(find_eigenvalue(E, dx, x, V)) for E in energies])
    
    energy_ranges = []
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1]:
            energy_ranges.append((energies[i], energies[i + 1]))
    
    return energy_ranges

def get_eigenvalues(V, x, dx, energy_min, energy_max, num_scan_points=100):
    energy_ranges = scan_energy_ranges(V, x, dx, energy_min, energy_max, num_scan_points)
    eigenvalues = []
    for E1, E2 in energy_ranges:
        E = brentq(find_eigenvalue, E1, E2, args=(dx, x, V))
        eigenvalues.append(E)
    return np.array(eigenvalues)


def normalize(psi, dx):
    return psi / np.sqrt(np.sum(psi[:-(psi.shape[0]//10)]**2) * dx) # cut last 10% due to error


def get_eigenfunction(V, x, dx, E):
    psi = normalize(numerov(E, dx, x, V), dx)
    def eigenfunction(z):
        index = int((z - x[0]) // dx)
        return psi[index]
    return eigenfunction

