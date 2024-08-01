from typing import Callable
import numpy as np
import vegas

class PIMCSingleParticle1DNonRel:
    """
    Class for calculating path integrals with Monte Carlo of 1 dimensional non-relativistic single-particle quantum system (such as the quantum harmonic oscillator)

    Attributes:
        V (Callable[float, float]): The potential characterizing the system.
        a (float): Temporal lattice spacing.
        N (int): Number of temporal slices.
        m (float): Mass of the particle.
    """

    def __init__(self, V: Callable, a: float, N: int, m: float=1.0):
        """
        Args:
            V (Callable[float, float]): The potential characterizing the system.
            a (float): Temporal lattice spacing.
            N (int): Number of temporal slices.
            m (float, optional): Mass of the particle. Default is $1.0$.        
        """
        self.V = V
        self.a = a
        self.N = N
        self.m = m
        
        self.A = (m/(2*np.pi*a))**(N/2) # Normalization factor of path integral in single particle non-relativistic 1 dimensional case

    def S_lat(self, path: np.ndarray):
        """
        Args:
            path (numpy.ndarray): Array of N positions representing the path.

        Returns:
            float: the value of the action for the specified path.
        """
        S = 0
        for j in range(self.N):
            S += self.m/(2*self.a)*(path[j+1] - path[j])**2 + self.a*self.V(path[j])
        return S
    

    def _integrand_factory(self, x: float):
        def integrand(path_var: np.ndarray):
            path = np.insert(path_var, 0, x)
            path = np.append(path, x)
            return self.A * np.exp(-self.S_lat(path))
        return integrand
    

    def compute_propagator(self, x: float, lower_bound=-5.0, upper_bound=5.0, nitn_tot=30, nitn_discarded=10, neval=2500):
        """
        Computes the propagator $$ \\langle x | e^{-\hat{H}T} | x \\rangle $$ where $T=Na$ using the `vegas` library, which implements the Monte Carlo estimation of multidimensional integrals.

        Args:
            x (float): Argument of the propagator.
            lower_bound (float, optional): Lower integration bound for each variable in the path. Default $5.0$.
            upper_bound (float, optional): Upper integration bound for each variable in the path. Default $-5.0$.
            nitn_tot (int): Total number of iterations of `vegas` algorithm to estimate the integral.
            nitn_discarded (int): Number of iterations of `vegas` algorithm to be discarded at the beginning, to let `vegas` adapt to the integrand without polluting the error analysis.
            neval (int): Number of Monte Carlo evaluations of the integral in each iteration of `vegas` algorithm.

        Returns:
            vegas.RAvg: The `vegas` result of the integration procedure. See https://vegas.readthedocs.io/en/latest/vegas.html#vegas.RAvg
        """
        domain = [[lower_bound, upper_bound]]*(self.N-1)
        integrator = vegas.Integrator(domain)
        integrator(self._integrand_factory(x), nitn=nitn_discarded, neval=neval)
        result = integrator(self._integrand_factory(x), nitn=nitn_tot-nitn_discarded, neval=neval)
        return result
    
    def get_T(self):
        return self.N * self.a