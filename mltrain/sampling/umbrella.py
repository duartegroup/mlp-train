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
                 rxn_coords: np.ndarray,
                 bias_e:     np.ndarray,
                 refs:       np.ndarray,
                 n_points:   int):
        """
        Umbrella Window

        -----------------------------------------------------------------------
        Arguments:

            rxn_coords: Values of the sampled reaction coordinate (q)

            bias_e: Values of the bias across the reaction coordinate for a
                    given window kappa and reference

            refs: Values of the reference for every window

            n_points: Number of points sampled in each window

        """
        bins = np.linspace(refs[0], refs[-1], num=n_points+1)
        self.hist, _ = np.histogram(rxn_coords, bins=bins)

        self.bias_e = bias_e
        self.free_energy = 0.0

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
        self.temp:              Optional[float] = None
        self._fitted_gaussians: List[_FittedGaussian] = []

        self.windows:           List = []
        self.prob_dist:         Optional[np.ndarray] = None
        self.n_points:          Optional[int] = None

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

        self.temp = temp

        self._set_reference_values(traj, n_windows, init_ref, final_ref)

        combined_traj = ConfigurationSet()
        for idx, ref in enumerate(self.refs):

            logger.info(f'Running US window {idx+1} with ζ={ref:.2f} Å and '
                        f'κ = {self.kappa:.3f} eV / Å^1')

            bias = Bias(zeta_func=self.zeta_func, kappa=self.kappa,
                        reference=ref)

            win_traj = _run_individual_window(self._best_init_frame(bias,
                                                                    traj),
                                              mlp,
                                              temp,
                                              interval,
                                              dt,
                                              bias=bias,
                                              **kwargs)

            self.n_points = len(win_traj)

            q_points = np.linspace(self.refs[0], self.refs[-1],
                                   num=self.n_points)

            self.windows.append(_Window(rxn_coords=self.zeta_func(win_traj),
                                        bias_e=bias.bias_over_range(q_points),
                                        refs=self.refs,
                                        n_points=self.n_points))

            gaussian = self._fit_gaussian(self.zeta_func(win_traj))
            self._fitted_gaussians.append(gaussian)

            combined_traj = combined_traj + win_traj

        plt.close()
        combined_traj.save(filename='combined_windows.xyz')

        return None

    def _plot_free_energy(self, units='kcal mol-1'):
        """Plots the free energy against the reaction coordinate"""
        free_energies = self.free_energies

        if units.lower() == 'ev':
            pass

        elif units.lower() == 'kcal mol-1':
            free_energies *= 23.060541945329334   # eV -> kcal mol-1

        elif units.lower() == 'kj mol-1':
            free_energies *= 96.48530749925793   # eV -> kJ mol-1

        else:
            raise ValueError(f'Unknown energy units: {units}')

        rel_free_energies = free_energies - min(free_energies)
        zetas = np.linspace(self.refs[0], self.refs[-1], num=self.n_points)

        plt.plot(zetas, rel_free_energies, color='k')

        with open(f'free_energy.txt', 'w') as outfile:
            for ref, free_energy in zip(zetas, rel_free_energies):
                print(ref, free_energy, file=outfile)

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('ΔG / kcal mol$^{-1}$')

        plt.savefig('free_energy.pdf')
        plt.close()
        return None

    @property
    def free_energies(self) -> np.ndarray:
        """
        Free energies at each point along the profile, eqn. 8.6.5 in Tuckermann

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): A(ζ)
        """
        return - (1.0 / self.beta) * np.log(self.prob_dist)

    @property
    def beta(self) -> float:
        """
        β = 1 / (k_B T)

        -----------------------------------------------------------------------
        Returns:
            (float): β in units of eV^-1
        """
        k_b = 8.617333262E-5  # Boltzmann constant in eV / K
        return 1.0 / (k_b * self.temp)

    def wham(self,
             tol:            float = 1E-3,
             max_iterations: int = 10000
             ) -> None:
        """
        Construct an unbiased distribution (on a grid) from a set of windows

        -----------------------------------------------------------------------
        Arguments:

            tol: Tolerance on the convergence

            max_iterations: Maximum number of WHAM iterations to perform
        """
        beta = self.beta

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
        self._plot_free_energy()

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
