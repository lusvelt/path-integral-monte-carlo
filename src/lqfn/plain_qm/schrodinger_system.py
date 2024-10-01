"""
This module contains code for modelling non-relativistic single particle 1-dimensional quantum systems, called "Schrodinger systems".
"""

from typing import Callable, List, Tuple
import concurrent.futures
import numpy as np
from qmsolve import Hamiltonian, SingleParticle
from qmsolve.eigenstates import Eigenstates
import vegas
from numba import njit
from .. import utils
from . import metropolis


class SchrodingerSystem:
    """
    This class models a non relativistic quantum system composed by a single particle in one spatial dimension, providing methods for computing relevant physical quantities using different numerical methods implemented elsewhere.

    The standard modelization of such a system is given by the Schrodinger equation, for which a potential $V$ and the mass of the particle $m$ should be given. In addition, we assume that the initial time is $t_i = 0$ and the final time is $t_f = T$.

    Attributes:
        m (float): Mass of the particle.
        T (float): Total evolution time of the system.
        N (int): Number of temporal lattice points for lattice calculations.
        box (Tuple[float, float]): Extremal spatial points `(x_min, x_max)` for the particle box (set to high value for infinite box).
        V (Callable[float] -> float | None): The potential characterizing the system.
        S_per_timeslice (Callable[np.ndarray, int] -> float | None): A functional taking as input an integer $j$ and a path, and returning the contribution of the $j$-th time instant of the path to the action.
    """

    def __init__(
        self,
        T: float,
        m: float = 1.0,
        N: int = 100,
        box: Tuple = (-100.0, 100.0),
        V: Callable | None = None,
        S_per_timeslice: Callable | None = None,
    ):
        """
        Either V or S_per_timeslice must be specified.

        Args:
            T (float): Total evolution time of the system.
            m (float, optional): Mass of the particle. Default is $1.0$.
            N (int): Number of temporal lattice points for lattice calculations.
            box (Tuple[float, float]): Extremal spatial points `(x_min, x_max)` for the particle box (set to high value w.r.t. characteristic length scale of the system for infinite box).
            V (Callable[float] -> float | None, optional): The potential characterizing the system.
            S_per_timeslice (Callable[np.ndarray, int, float] -> float | None, optional): A functional taking as input an integer $j$, a path and the lattice spacing, and returning the contribution of the $j$-th time instant of the path to the action.
        """
        assert (V is not None) or (S_per_timeslice is not None)
        self.T = T
        self.m = m
        self.N = N
        self.box = np.array(list(box))
        self.V = V

        # Since a is fixed at this point, saturate the a parameter of the function that computes the action
        @njit
        def S(j, x):
            return S_per_timeslice(j, x, T / N)

        self.S_per_timeslice = S
        self._eigenstates = None  # This attribute will eventually be filled when Schrodinger equation will be solved
        self._eigenstates_N = None  # This saves the precision by which the Schrodinger equation results have been stored

    @property
    def a(self) -> float:
        """
        float: Temporal interval spacing for lattice calculations.
        """
        return self.T / self.N

    @property
    def A(self) -> float:
        """
        float: Normalization constant of the path integral
        """
        return (self.m / (2 * np.pi * self.a)) ** (self.N / 2)

    def S_lat(self, path: np.ndarray) -> float:
        """
        Computes the action of a path defined on the lattice characterized by $N$ temporal slices.

        Args:
            path (numpy.ndarray[N+1]): Array of N+1 positions representing the path.

        Returns:
            float: the value of the action for the specified path.
        """
        S = 0
        for j in range(self.N):
            S += self.m / (2 * self.a) * (path[j + 1] - path[j]) ** 2 + self.a * self.V(path[j])
        return S

    # This is a factory function that generates a function, which will be the integrand for vegas.
    # The integrand takes the middle points of the path, and has the end points of the path fixed at x
    def _integrand_factory(self, x: float) -> Callable:
        def integrand(path_var: np.ndarray):
            path = np.insert(path_var, 0, x)
            path = np.append(path, x)
            return self.A * np.exp(-self.S_lat(path))

        return integrand

    def compute_propagator_pimc(
        self, x: np.ndarray | float, lower_bound=-5.0, upper_bound=5.0, nitn_tot=30, nitn_discarded=10, neval=2500, max_workers=16
    ) -> vegas.RAvg | List[vegas.RAvg]:
        """
        Computes the propagator $\\bra{x} e^{-\\hat{H}T} \\ket{x}$ through the discretized path integral formula, using the `vegas` library, which implements the Monte Carlo estimation of multidimensional integrals. See section 2.1 of Lepage's "Lattice QCD for novices" paper for more information.

        Parallelism is used to compute more points at the same time.

        Args:
            x (float | np.ndarray): Argument of the propagator (or array of arguments).
            lower_bound (float, optional): Lower integration bound for each variable in the path. Default $-5.0$.
            upper_bound (float, optional): Upper integration bound for each variable in the path. Default $5.0$.
            nitn_tot (int): Total number of iterations of `vegas` algorithm to estimate the integral.
            nitn_discarded (int): Number of iterations of `vegas` algorithm to be discarded at the beginning, to let `vegas` adapt to the integrand without polluting the error analysis.
            neval (int): Number of Monte Carlo evaluations of the integral in each iteration of `vegas` algorithm.
            max_workers (int): Maximum number of threads to use for the calculation
        Returns:
            vegas.RAvg: The `vegas` result of the integration procedure. See https://vegas.readthedocs.io/en/latest/vegas.html#vegas.RAvg
            List[vegas.RAvg]: The list of results for each separate integrand.
        """
        domain = [[lower_bound, upper_bound]] * (self.N - 1)

        # Define the job for the threads. Each thread computes a single point of the propagator.
        def get_single_value(y):
            integrator = vegas.Integrator(domain)
            f = self._integrand_factory(y)
            integrator(f, nitn=nitn_discarded, neval=neval)
            result = integrator(f, nitn=nitn_tot - nitn_discarded, neval=neval)
            return result

        if isinstance(x, float):
            return get_single_value(x)
        else:
            assert isinstance(x, np.ndarray)
            # Launch threads execution. The function returns when all threads have finished, and the order is kept in the results array.
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(get_single_value, z) for z in x]
                results = [job.result() for job in futures]
                return results

    def compute_propagator_from_ground_state(
        self,
        x: np.ndarray,
        ground_wavefunction: Callable | None = None,
        ground_energy: float | None = None,
    ) -> np.ndarray:
        """
        Computes the values of the propagator $\\bra{x} e^{-\\hat{H}T} \\ket{x}$

        in the approximation of large $T$, using the formula:

        $$\\bra{x} e^{-\\hat{H}T} \\ket{x} \\approx e^{-E_0 T} {\\lvert\\braket{x | E_0}\\rvert}^2 $$

        """
        if (ground_wavefunction is None) or (ground_energy is None):
            sym_box_half = np.max(np.abs(self.box))
            schrodinger_N = utils.get_extended_linspace_size(x, extent=2 * sym_box_half)
            eigenstates = self.solve_schrodinger(N=schrodinger_N, max_states=1)
            ground_wavefunction = eigenstates.array[0]
            ground_energy = eigenstates.energies[0]
            idxs = utils.get_linspace_idxs_within(x, x_min=-sym_box_half)
        else:
            ground_wavefunction = ground_wavefunction(x)
            idxs = np.arange(x.shape[0])
        assert (ground_wavefunction is not None) and (ground_energy is not None)
        return ground_wavefunction[idxs] ** 2 * np.exp(-1 * ground_energy * self.T)

    def solve_schrodinger(self, N: int, max_states: int) -> Eigenstates:
        """
        Solves the Schrodinger equation using the `qmsolve` library.

        Args:
            N (int): Resolution (number of points) of the solution array.
            max_states (int): Number of eigenstates to compute.

        Returns:
            qmsolve.eigenstates.Eigenstates: Object containing the eigenstates (`Eigenstates.array`) and energy levels (`Eigenstates.energies`) in atomic units.
        """
        ENERGY_HARTREE_EV = 27.211386245988
        particle = SingleParticle(m=self.m)

        def potential(particle):
            return self.V(particle.x)

        H = Hamiltonian(particle, potential, N, extent=2 * np.max(self.box), spatial_ndim=1)
        eigenstates = H.solve(max_states)
        eigenstates.energies = eigenstates.energies / ENERGY_HARTREE_EV
        return eigenstates

    def get_delta_E_schrodinger(self, N: int = 100000) -> float:
        """
        Computes $\\Delta E$ (that is, the first excitation energy of the system multiplied by the lattice spacing $a$) by solving the Schrodinger equation.

        Args:
            N (int): Resolution (number of points) of the eigenfunction (affects the precision of $\\Delta E)

        Returns:
            float: The computed value of $\\Delta E$
        """
        # Save eigenstates in a local variable to avoid computing them each time
        # Only compute them if precision changes or if they have never been computed
        if self._eigenstates is None or self._eigenstates_N != N:
            self._eigenstates_N = N
            self._eigenstates = self.solve_schrodinger(N, 2)
        return (self._eigenstates.energies[1] - self._eigenstates.energies[0]) * self.a

    def compute_delta_E_pimc(
        self,
        functional: Callable,
        N_cf: int,
        N_cor: int,
        eps: float,
        thermalization_its: int = 5,
        N_copies: int = 1,
        bin_size: int = 1,
        N_points: int = None,
    ) -> np.ndarray:
        """
        Computes the $\\Delta E$ from a correlation function as in (38), that is:
        $$\\Delta E_n = \\log (G_n / G_{n+1})$$
        It also computes the error on $\\Delta E_n$.
        To compute the correlation function, the Monte Carlo metropolis algorithm is used.

        Two differenth methods can be used:
        - If N_copies == 1: a single path integral is performed to compute the correlation function, and then $\\Delta E$ is computed straightforwardly. The error is propagated from the correlator using:
        $$\\delta \\Delta E_n = \\sqrt { \\left({\\frac{\\partial \\Delta E_n}{\\partial G_n}} \\right)^2  \\delta G_{n}^2 +  \\left({\\frac{\\partial \\Delta E_n}{\\partial G_{n+1}}} \\right)^2 \\delta G_{n+1}^2 } = \\sqrt{ \\left( \\frac{\\delta G_n}{G_n} \\right)^2 + \\left(\\frac{\\delta G_{n+1}}{G_{n+1}} \\right)^2 } $$
        - If 1 < N_copies <= N_cf: bootstrap procedure is used both for computing $\\Delta E$ and for the error calculation. See Lepage's paper for more info.

        Args:
            functional (Callable[np.ndarray, int] -> float): Discretized correlation functional that takes a path and a time index and returns the value of the correlator.
            N_cf (int): Total number of samples contributing to be saved during the process for computing the path integral average.
            N_cor (int): Number of path updates before picking each sample.
            eps (float): $\\epsilon$ parameter for the update of the path.
            thermalization_its (int, optional): Number of samples to be discarded at the beginning to let the procedure thermalize. Default is 5.
            N_copies (int, optional): Number of copies to be used for the bootstrap procedure.
            bin_size (int, optional): Number of samples to be averaged in a single bin. Default is 1.
            N_points (int, optional): Number of lattice points to perform the calculation for. Default is the max value, specified in the constructor. One may choose to compute $\\Delta E$ only for a few starting points to avoid asting computation time.
        Returns:
            np.ndarray: Series of computed N-1 $\\Delta E$s.
            np.ndarray: Series of computed errors of $\\Delta E$s.
        """
        assert N_copies <= N_cf
        if N_points is None:
            N_points = self.N
        else:
            assert 0 < N_points <= self.N

        propagator = np.zeros((N_copies, N_points), dtype=np.float64)
        propagator_err = np.zeros((N_copies, N_points), dtype=np.float64)

        propagator, propagator_err = metropolis.compute_path_integral_average(
            functional,
            self.S_per_timeslice,
            self.N,
            N_cf,
            N_cor,
            eps,
            thermalization_its,
            N_copies,
            bin_size,
            N_points,
        )

        # Now we have a pool of samples, and we can compute \Delta E and its error
        delta_E = np.zeros((N_copies, N_points - 1))
        for i in range(N_copies):
            for n in range(0, N_points - 1):
                delta_E[i, n] = np.log(np.abs(propagator[i, n] / propagator[i, n + 1]))
        delta_E_avg = delta_E.mean(axis=0)

        if N_copies == 1:
            error_delta_E = np.zeros(N_points - 1)
            for n in range(0, N_points - 1):
                error_delta_E[n] = np.sqrt(((propagator_err[0, n] / propagator[0, n]) ** 2 + (propagator_err[0, n + 1] / propagator[0, n + 1]) ** 2))
        else:
            error_delta_E = delta_E.std(axis=0)
        return delta_E_avg, error_delta_E
