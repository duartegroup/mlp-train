import mltrain
import numpy as np
import matplotlib.pyplot as plt
from mltrain.sampling.bias import Bias
from mltrain.sampling.reaction_coord import ReactionCoordinate
from mltrain.configurations import ConfigurationSet
from mltrain.sampling.md import run_mlp_md
from mltrain.log import logger
from typing import Optional, List, Callable
from scipy.optimize import curve_fit


def _run_individual_window(frame, mlp, temp, interval, dt, bias, **kwargs):
    """Runs an individual umbrella sampling window. Adaptive sampling to
    be implemented"""

    traj = run_mlp_md(configuration=frame,
                      mlp=mlp,
                      temp=temp,
                      dt=dt,
                      interval=interval,
                      bias=bias,
                      **kwargs)

    return traj


class _Window:
    """Contains the attributes belonging to an US window used for WHAM or UI"""

    def __init__(self,
                 bias_e: np.ndarray,
                 rxn_coords: np.ndarray):
        """
        Umbrella Window

        -----------------------------------------------------------------------
        Arguments:
            bias_e: Value of the bias for each frame in a window

            rxn_coords: Values of the reaction coordinate (q)

        """
        self.bias_e = bias_e
        self.rxn_coords = rxn_coords

        self.hist, _ = np.histogram(bias_e, bins=self.bins)
        self.free_energy = 0.0

    @property
    def bins(self) -> np.ndarray:
        """Bin edges to histogram into"""
        return np.linspace(np.min(self.rxn_coords), np.max(self.rxn_coords),
                           num=len(self.rxn_coords) + 1)

    @property
    def n(self) -> int:
        """Number of samples in this window"""
        return int(np.sum(self.hist))


class UmbrellaSampling:
    """
    Umbrella sampling class for generating pulling simulation, running
    umbrella sampling windows and running WHAM or umbrella integration.
    """

    def __init__(self,
                 zeta_func: 'mltrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa:      float):
        """
        Umbrella sampling to predict free energy using an mlp under a harmonic
        bias:

            ω = κ/2 (ζ(r) - ζ_ref)^2

        where ω is the bias in a particular window, ζ a function that takes in
        nuclear positions (r) and returns a scalar and ζ_ref the reference
        value of the reaction coordinate in that particular window.

        -----------------------------------------------------------------------
        Arguments:

            zeta_func: Reaction coordinate, as the function of atomic positions

            kappa: Value of the spring_const, κ, used in umbrella sampling
        """

        self.kappa:             float = kappa
        self.zeta_func:         Callable = zeta_func
        self.refs:              Optional[np.ndarray] = None
        self._fitted_gaussians: List[_FittedGaussian] = []

        self.windows:           List = []
        self.free_energy:       Optional[np.ndarray] = None
        self.prob_dist:         Optional[np.ndarray] = None
        self.n_points:          int = 0

    @staticmethod
    def _best_init_frame(bias, traj):
        """Find the frames whose bias value is the lowest, i.e. has the
        closest reaction coordinate to the desired"""

        min_e_idx = np.argmin([bias(frame.ase_atoms) for frame in traj])

        return traj[min_e_idx]

    def _set_reference_values(self, traj, num, init_ref, final_ref) -> None:
        """Set the values of the reference for each window, if the
        initial and final reference values of the reaction coordinate are None
        then use the values in the start or end of the trajectory"""

        if init_ref is None:
            init_ref = self.zeta_func(traj[0])

        if final_ref is None:
            final_ref = self.zeta_func(traj[-1])

        self.refs = np.linspace(init_ref, final_ref, num)
        return None

    def _fit_gaussian(self, data) -> '_FittedGaussian':
        """Fit a Gaussian to a set of data"""
        gaussian = _FittedGaussian()

        min_x = min(self.refs) * 0.9
        max_x = max(self.refs) * 1.1

        x_range = np.linspace(min_x, max_x, 500)

        hist, bin_edges = np.histogram(data, density=False, bins=500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

        initial_guess = [1.0, 1.0, 1.0]
        gaussian.params, _ = curve_fit(gaussian.value, bin_centres, hist,
                                       p0=initial_guess,
                                       maxfev=10000)

        plt.plot(bin_centres, hist, alpha=0.1)
        plt.plot(x_range, gaussian(x_range))

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')

        plt.savefig('fitted_data.pdf')

        return gaussian

    def run_umbrella_sampling(self,
                              traj:      'mltrain.ConfigurationSet',
                              mlp:       'mltrain.potentials._base.MLPotential',
                              temp:      float,
                              interval:  int,
                              dt:        float,
                              init_ref:  Optional[float] = None,
                              final_ref: Optional[float] = None,
                              n_windows: int = 10,
                              **kwargs
                              ) -> None:
        """
        Run umbrella sampling across n_windows, fitting Gaussians to the
        sampled values of the reaction coordinate.

        -----------------------------------------------------------------------
        Arguments:
            traj: Trajectory from which to initialise the umbrella over, e.g.
                  a 'pulling' trajectory that has sufficient sampling of a
                  range f reaction coordinates

            mlp: Machine learnt potential

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive
            
            interval: (int) Interval between saving the geometry
            
            dt: (float) Time-step in fs
            
            init_ref: (float | None) Value of reaction coordinate in Å for
                       first window
            
            final_ref: (float | None) Value of reaction coordinate in Å for
                       first window
            
            n_windows: (int) Number of windows to run in the umbrella sampling

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units

        Returns:
            None:
        """
        if temp <= 0:
            raise ValueError('Temperature must be positive and non-zero for '
                             'umbrella sampling')

        self._set_reference_values(traj, n_windows, init_ref, final_ref)

        combined_traj = ConfigurationSet()
        for idx, ref in enumerate(self.refs):

            logger.info(f'Running US window {idx} with ζ={ref:.2f} Å and '
                        f'κ = {self.kappa:.5f} eV / Å^2')

            bias = Bias(zeta_func=self.zeta_func, kappa=self.kappa,
                        reference=ref)

            traj = _run_individual_window(self._best_init_frame(bias, traj),
                                          mlp,
                                          temp,
                                          interval,
                                          dt,
                                          bias=bias,
                                          **kwargs)

            self.windows.append(_Window(bias(traj), self.zeta_func(traj)))
            self.n_points = len(traj)

            gaussian = self._fit_gaussian(self.zeta_func(traj))
            self._fitted_gaussians.append(gaussian)

            combined_traj = combined_traj + traj

        combined_traj.save(filename='combined_windows.xyz')

        return None

    def wham(self,
             beta=1.0,
             tol=1E-3,
             max_iterations=10000
             ) -> None:
        """
        Construct an unbiased distribution (on a grid) from a set of windows

        -----------------------------------------------------------------------
        Arguments:
            beta: 1 / k_B T

            tol: Tolerance on the convergence

            max_iterations: Maximum number of WHAM iterations to perform
        """
        self.free_energy = np.zeros(self.n_points)

        # Uniform probability distribution starting point
        self.prob_dist = np.ones(self.n_points) / self.n_points

        p_prev = np.inf * np.ones(self.n_points)  # Start with P(q)^(-1) = ∞
        prob_dist = self.prob_dist

        def converged():
            return np.max(np.abs(p_prev - prob_dist)) < tol

        for _ in range(max_iterations):

            prob_dist = (sum(w_k.hist for w_k in self.windows)
                 / sum(w_k.n * np.exp(beta * (w_k.free_energy - w_k.bias_e))
                       for w_k in self.windows))

            for w_k in self.windows:
                w_k.free_energy = (-(1.0/beta)
                         * np.log(np.sum(prob_dist * np.exp(-w_k.bias_e * beta))))

            if converged():
                break

            p_prev = prob_dist

        self.prob_dist = prob_dist
        return None


class _FittedGaussian:

    def __init__(self,
                 a: float = 1.0,
                 b: float = 1.0,
                 c: float = 1.0):
        """
        Gaussian defined by three parameters:

        a * exp(-(x - b)^2 / (2 * c^2))
        """
        self.params = a, b, c

    def __call__(self, x):
        return self.value(x, *self.params)

    @staticmethod
    def value(x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2. * c**2))
