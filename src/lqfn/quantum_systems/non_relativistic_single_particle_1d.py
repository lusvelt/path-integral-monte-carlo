

from typing import Callable, List
import numpy as np
import vegas


class NonRelativisticSingleParticle1D:
    """
    This class models a non relativistic quantum system composed by a single particle in one spatial dimension, providing methods for computing relevant physical quantities using different numerical methods implemented elsewhere.

    The standard modelization of such a system is given by the Schrodinger equation, for which a potential $V$ and the mass of the particle $m$ should be given. In addition, we assume that the initial time is $t_i = 0$ and the final time is $t_f = T$.

    Attributes:
        V (Callable[float, float]): The potential characterizing the system.
        m (float): Mass of the particle.
        T (float): Total evolution time of the system.
        N (int): Number of temporal lattice points for lattice calculations.
        a (float): Temporal interval spacing for lattice calculations.
    """

    def __init__(self, V: Callable, T: float, m: float=1.0, N: int = 100):
        """
        Args:
            V (Callable[float, float]): The potential characterizing the system.
            T (float): Total evolution time of the system.
            m (float, optional): Mass of the particle. Default is $1.0$.
            N (int): Number of temporal lattice points for lattice calculations.
        """
        self.V = V
        self.T = T
        self.m = m
        self.N = N
        
    @property
    def a(self):
        return self.T / self.N

    @property
    def _A(self): # Normalization constant of path integral
        return (self.m/(2*np.pi*self.a))**(self.N/2)


    def S_lat(self, path: np.ndarray) -> float:
        """
        Computes the action of a path defined on the lattice characterized by $N$ temporal slices.

        Args:
            path (numpy.ndarray[N]): Array of N positions representing the path.

        Returns:
            float: the value of the action for the specified path.
        """
        S = 0
        for j in range(self.N):
            S += self.m/(2*self.a)*(path[j+1] - path[j])**2 + self.a*self.V(path[j])
        return S
    

    def _integrand_factory(self, x: float) -> Callable:
        def integrand(path_var: np.ndarray):
            path = np.insert(path_var, 0, x)
            path = np.append(path, x)
            return self._A * np.exp(-self.S_lat(path))
        return integrand
    

    def compute_propagator_pimc(self, x: np.ndarray | float, lower_bound=-5.0, upper_bound=5.0, nitn_tot=30, nitn_discarded=20, neval=2500) -> vegas.RAvg | List[vegas.RAvg]:
        """
        Computes the propagator $\\bra{x} e^{-\\hat{H}T} \\ket{x}$ through the discretized path integral formula, using the `vegas` library, which implements the Monte Carlo estimation of multidimensional integrals. See section 2.1 of Lepage's "Lattice QCD for novices" paper for more information.

        Args:
            x (float | np.ndarray): Argument of the propagator (or array of arguments).
            lower_bound (float, optional): Lower integration bound for each variable in the path. Default $-5.0$.
            upper_bound (float, optional): Upper integration bound for each variable in the path. Default $5.0$.
            nitn_tot (int): Total number of iterations of `vegas` algorithm to estimate the integral.
            nitn_discarded (int): Number of iterations of `vegas` algorithm to be discarded at the beginning, to let `vegas` adapt to the integrand without polluting the error analysis.
            neval (int): Number of Monte Carlo evaluations of the integral in each iteration of `vegas` algorithm.

        Returns:
            vegas.RAvg: The `vegas` result of the integration procedure. See https://vegas.readthedocs.io/en/latest/vegas.html#vegas.RAvg
            List[vegas.RAvg]: The list of results for each separate integrand.
        """
        # TODO: see if another level of abstraction can be inserted, which handles the setup of vegas integration
        domain = [[lower_bound, upper_bound]]*(self.N-1)
        results = []
        if not isinstance(x, np.ndarray):
            x = np.array([x])
        results = []
        for x_i in x:
            integrator = vegas.Integrator(domain)
            f = self._integrand_factory(x_i)
            integrator(f, nitn=nitn_discarded, neval=neval)
            result = integrator(f, nitn=nitn_tot-nitn_discarded, neval=neval)
            results.append(result)
        if len(results) == 1:
            return results[0]
        else:
            return results
    

    def compute_propagator_from_ground_state(self, x: np.ndarray, ground_wavefunction: Callable | None = None, ground_energy: float | None = None) -> np.ndarray:
        """
        Computes the values of the propagator $\\bra{x} e^{-\\hat{H}T} \\ket{x}$
        
        in the approximation of large $T$, using the formula:
        
        $$\\bra{x} e^{-\\hat{H}T} \\ket{x} \\approx e^{-E_0 T} {\\lvert\\braket{x | E_0}\\rvert}^2 $$

        """
        if (ground_wavefunction is None) or (ground_energy is None):
            # TODO: call Schrodinger solver procedure
            raise NotImplementedError()
        assert (ground_wavefunction is not None) and (ground_energy is not None)
        return ground_wavefunction(x)**2 * np.exp(-ground_energy*self.T)
