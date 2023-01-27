import os
import time
import mlptrain
import numpy as np
import matplotlib.pyplot as plt
from mlptrain.sampling.bias import Bias
from mlptrain.sampling.reaction_coord import ReactionCoordinate, DummyCoordinate
from mlptrain.configurations import ConfigurationSet
from mlptrain.sampling.md import run_mlp_md
from mlptrain.config import Config
from mlptrain.log import logger
from typing import Optional, List, Callable, Tuple
from scipy.optimize import curve_fit
from multiprocessing import Pool


def _run_individual_window(frame, mlp, temp, interval, dt, bias, **kwargs):
    """Runs an individual umbrella sampling window. Adaptive sampling to
    be implemented"""

    traj = run_mlp_md(configuration=frame,
                      mlp=mlp,
                      temp=temp,
                      dt=dt,
                      interval=interval,
                      bias=bias,
                      umbrella=True,
                      **kwargs)

    return traj


class _Window:
    """Contains the attributes belonging to an US window used for WHAM or UI"""

    def __init__(self,
                 obs_zetas: np.ndarray,
                 bias: 'mlptrain.Bias'):
        """
        Umbrella Window

        -----------------------------------------------------------------------
        Arguments:

            obs_zetas: Values of the sampled (observed) reaction coordinate
                       ζ_i for this window (i)

            bias: Bias function, containing a reference value of ζ in this
                  window and its associated spring constant
        """
        self._bias = bias
        self._obs_zetas = obs_zetas

        self.fitted_gaussian: Optional[_FittedGaussian] = None

        self.bias_energies: Optional[np.ndarray] = None
        self.hist:          Optional[np.ndarray] = None

        self.free_energy = 0.0

    def bin(self,
            zetas: np.ndarray) -> None:
        """
        Bin the observed reaction coordinates in this window into an a set of
        bins, defined by the array of bin centres (zetas)

        -----------------------------------------------------------------------
        Arguments:
            zetas: Discretized reaction coordinate
        """

        bins = np.linspace(zetas[0], zetas[-1], num=len(zetas)+1)
        self.hist, _ = np.histogram(self._obs_zetas, bins=bins)

        self.bias_energies = (self._bias.kappa/2) * (zetas - self._bias.ref)**2

        return None

    @property
    def n(self) -> int:
        """Number of samples in this window"""
        if self.hist is None:
            raise ValueError('Cannot determine the number of samples - '
                             'window has not been binned')

        return int(np.sum(self.hist))

    @property
    def zeta_ref(self) -> float:
        """
        ζ_ref for this window

        -----------------------------------------------------------------------
        Returns:
            (float):
        """
        return self._bias.ref

    @classmethod
    def from_file(cls, filename: str) -> '_Window':
        """
        Load a window from a saved file

        -----------------------------------------------------------------------
        Arguments:
            filename:

        Returns:
            (mlptrain.sampling.umbrella._Window):
        """
        file_lines = open(filename, 'r', errors='ignore').readlines()
        header_line = file_lines.pop(0)            # Pop the first line

        ref_zeta = float(header_line.split()[0])   # Å
        kappa = float(header_line.split()[1])      # eV / Å^2

        obs_zeta = [float(line.split()[0]) for line in file_lines
                    if len(line.split()) > 0]

        window = cls(obs_zetas=np.array(obs_zeta),
                     bias=Bias(zeta_func=DummyCoordinate(),
                               kappa=kappa,
                               reference=ref_zeta))

        return window

    def save(self, filename: str) -> None:
        """
        Save this window to a file

        -----------------------------------------------------------------------
        Arguments:
            filename:
        """
        with open(filename, 'w') as out_file:
            print(self._bias.ref, self._bias.kappa, file=out_file)

            for zeta in self._obs_zetas:
                print(zeta, file=out_file)

        return None

    def _fit_gaussian(self, hist, bin_centres):
        """Fit a Gaussian to a histogram of data"""

        gaussian = _FittedGaussian()

        try:
            gaussian.params, _ = curve_fit(gaussian.value, bin_centres,
                                           hist,
                                           p0=[1.0, 1.0, 1.0],  # init guess
                                           maxfev=10000)

            if np.min(np.abs(bin_centres - gaussian.mean)) > 1.0:
                raise RuntimeError('Gaussian mean was not within the 1 Å of '
                                   'the ζ range')

        except RuntimeError:
            logger.error('Failed to fit a gaussian to this data')
            return None

        # Plot the fitted line in the same color as the histogram
        color = plt.gca().lines[-1].get_color()
        zetas = np.linspace(min(bin_centres), max(bin_centres), num=500)

        plt.plot(zetas, gaussian(zetas), c=color)

        self.fitted_gaussian = gaussian
        return None

    def plot(self,
             min_zeta:     float,
             max_zeta:     float,
             fit_gaussian: bool = True) -> None:
        """
        Plot this window along with a fitted Gaussian function if possible

        -----------------------------------------------------------------------
        Arguments:
            min_zeta:

            max_zeta:

            fit_gaussian:
        """
        hist, bin_edges = np.histogram(self._obs_zetas,
                                       density=False,
                                       bins=np.linspace(min_zeta - 0.1*abs(min_zeta),
                                                        max_zeta + 0.1*abs(max_zeta),
                                                        num=400))

        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2
        plt.plot(bin_centres, hist, alpha=0.1)

        if fit_gaussian:
            self._fit_gaussian(hist, bin_centres)

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('fitted_data.pdf')

        return None


class UmbrellaSampling:
    """
    Umbrella sampling class for generating pulling simulation, running
    umbrella sampling windows and running WHAM or umbrella integration.
    """

    def __init__(self,
                 zeta_func: 'mlptrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa:      float,
                 temp:       Optional[float] = None):
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

            kappa: Value of the spring constant, κ, used in umbrella sampling
        """

        self.kappa:             float = kappa                        # eV Å^-2
        self.zeta_func:         Callable = zeta_func                 # ζ(r)
        self.temp:              Optional[float] = temp               # K

        self.windows:           List[_Window] = []

    @staticmethod
    def _best_init_frame(bias, traj):
        """Find the frames whose bias value is the lowest, i.e. has the
        closest reaction coordinate to the desired"""
        if len(traj) == 0:
            raise RuntimeError('Cannot determine the best frame from a '
                               'trajectory with length zero')

        min_e_idx = np.argmin([bias(frame.ase_atoms) for frame in traj])

        return traj[min_e_idx]

    def _reference_values(self, traj, num, init_ref, final_ref) -> np.ndarray:
        """Set the values of the reference for each window, if the
        initial and final reference values of the reaction coordinate are None
        then use the values in the start or end of the trajectory"""

        if init_ref is None:
            init_ref = self.zeta_func(traj[0])

        if final_ref is None:
            final_ref = self.zeta_func(traj[-1])

        return np.linspace(init_ref, final_ref, num)

    def _no_ok_frame_in(self, traj, ref) -> bool:
        """
        Does there exist a good reference structure in a trajectory?
        defined by the minimum absolute difference in the reaction coordinate
        (ζ) observed in the trajectory and the target value

        -----------------------------------------------------------------------
        Arguments:
            traj: A trajectory containing structures
            ref: ζ_ref

        Returns:
            (bool):
        """
        return np.min(np.abs(self.zeta_func(traj) - ref)) > 0.5

    def run_umbrella_sampling(self,
                              traj: 'mlptrain.ConfigurationSet',
                              mlp: 'mlptrain.potentials._base.MLPotential',
                              temp:      float,
                              interval:  int,
                              dt:        float,
                              init_ref:  Optional[float] = None,
                              final_ref: Optional[float] = None,
                              n_windows: int = 10,
                              save_sep:  Optional[bool] = False,
                              **kwargs
                              ) -> None:
        """
        Run umbrella sampling across n_windows, fitting Gaussians to the
        sampled values of the reaction coordinate.

        *NOTE* will leave a dangling plt.figure open

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

            save_sep: (bool) If True saves trajectories of each window separately

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """
        if temp <= 0:
            raise ValueError('Temperature must be positive and non-zero for '
                             'umbrella sampling')

        start_umbrella = time.perf_counter()

        self.temp = temp
        zeta_refs = self._reference_values(traj, n_windows, init_ref, final_ref)

        # window_process.get() --> window_traj
        window_trajs = []
        window_processes = []
        biases = []

        n_processes = min(n_windows, Config.n_cores)
        logger.info(f'Running Umbrella Sampling with {n_windows} windows, '
                    f'{n_processes} windows are run in parallel')

        with Pool(processes=n_processes) as pool:

            for idx, ref in enumerate(zeta_refs):

                logger.info(f'Running US window {idx} with ζ_ref={ref:.2f} Å '
                            f'and κ = {self.kappa:.3f} eV / Å^2')

                bias = Bias(self.zeta_func, kappa=self.kappa, reference=ref)

                if self._no_ok_frame_in(traj, ref):
                    # Takes the trajectory of the previous window, .get() blocks
                    # the main process until the previous window finishes
                    _traj = window_processes[idx-1].get()
                else:
                    _traj = traj

                init_frame = self._best_init_frame(bias, _traj)

                window_process = pool.apply_async(func=_run_individual_window,
                                                args=(init_frame,
                                                      mlp,
                                                      temp,
                                                      interval,
                                                      dt,
                                                      bias),
                                                kwds=kwargs)
                window_processes.append(window_process)
                biases.append(bias)

            for window_process, bias in zip(window_processes, biases):

                window_traj = window_process.get()
                window = _Window(obs_zetas=self.zeta_func(window_traj),
                                 bias=bias)
                window.plot(min_zeta=min(zeta_refs),
                            max_zeta=max(zeta_refs),
                            fit_gaussian=True)

                self.windows.append(window)
                window_trajs.append(window_traj)

        if save_sep:
            os.mkdir('trajectories')
            for idx, window_traj in enumerate(window_trajs):
                window_traj.save(filename=f'trajectories/window_{idx}.xyz')

        else:
            combined_traj = ConfigurationSet()
            for window_traj in window_trajs:
                combined_traj += window_traj

            combined_traj.save(filename='combined_windows.xyz')

        finish_umbrella = time.perf_counter()
        logger.info('Umbrella sampling done in '
                    f'{(finish_umbrella - start_umbrella) / 60:.1f} m')

        return None

    def free_energies(self, prob_dist) -> np.ndarray:
        """
        Free energies at each point along the profile, eqn. 8.6.5 in Tuckerman

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): A(ζ)
        """
        return - (1.0 / self.beta) * np.log(prob_dist)

    @property
    def zeta_refs(self) -> Optional[np.ndarray]:
        """
        Array of ζ_ref for each window

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray(float) | None):
        """
        if len(self.windows) == 0:
            return None

        return np.array([w_k.zeta_ref for w_k in self.windows])

    @property
    def beta(self) -> float:
        """
        β = 1 / (k_B T)

        -----------------------------------------------------------------------
        Returns:
            (float): β in units of eV^-1
        """
        if self.temp is None:
            raise ValueError('Cannot calculate β without a defined temperature'
                             ' please set .temp')

        k_b = 8.617333262E-5  # Boltzmann constant in eV / K
        return 1.0 / (k_b * self.temp)

    def wham(self,
             tol:            float = 1E-3,
             max_iterations: int = 100000,
             n_bins:         int = 100
             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct an unbiased distribution (on a grid) from a set of windows

        -----------------------------------------------------------------------
        Arguments:

            tol: Tolerance on the convergence

            max_iterations: Maximum number of WHAM iterations to perform

            n_bins: Number of bins to use in the histogram (minus one) and
                    the number of reaction coordinate values plotted and
                    returned

        Returns:
            (np.ndarray, np.ndarray): Tuple containing the reaction coordinate
                                      and values of the free energy
        """
        beta = self.beta   # 1 / (k_B T)

        # Discretized values of the reaction coordinate
        zetas = np.linspace(self.zeta_refs[0], self.zeta_refs[-1], num=n_bins)

        for window in self.windows:
            window.bin(zetas=zetas)

        p = np.ones_like(zetas) / len(zetas)  # P(ζ) uniform distribution
        p_prev = np.inf * np.ones_like(p)     # Start with P(ζ)_(-1) = ∞

        def converged():
            return np.max(np.abs(p_prev - p)) < tol

        for iteration in range(max_iterations):

            # Equation 8.8.18 from Tuckerman, p. 343
            p = (sum(w_k.hist for w_k in self.windows)
                 / sum(w_k.n * np.exp(beta * (w_k.free_energy - w_k.bias_energies))
                       for w_k in self.windows))

            for w_k in self.windows:
                # Equation 8.8.19 from Tuckerman, p. 343
                w_k.free_energy = (-(1.0/beta)
                                   * np.log(np.sum(p * np.exp(-w_k.bias_energies * beta))))

            if converged():
                logger.info(f'WHAM converged in {iteration} iterations')
                break

            p_prev = p

        _plot_and_save_free_energy(free_energies=self.free_energies(p),
                                   zetas=zetas)
        return zetas, self.free_energies(p)

    def save(self, folder_name: str = 'umbrella') -> None:
        """
        Save the windows in this US to a folder containing each window as .txt
        files within in
        """

        if len(self.windows) is None:
            logger.error(f'Cannot save US to {folder_name} - had no windows')
            return None

        os.mkdir(folder_name)
        for idx, window in enumerate(self.windows):
            window.save(filename=os.path.join(folder_name, f'window_{idx}.txt'))

        return None

    def load(self, folder_name: str) -> None:
        """Load data from a set of saved windows"""

        if not os.path.isdir(folder_name):
            raise ValueError(f'Loading from a folder was not possible as '
                             f'{folder_name} is not a valid folder')

        for filename in os.listdir(folder_name):

            if filename.startswith('window_') and filename.endswith('.txt'):
                window = _Window.from_file(os.path.join(folder_name, filename))
                self.windows.append(window)

        return None

    @classmethod
    def from_folder(cls,
                    folder_name: str,
                    temp: float) -> 'UmbrellaSampling':
        """
        Create an umbrella sampling instance from a folder containing the
        window data

        -----------------------------------------------------------------------
        Arguments:
            folder_name:

            temp: Temperature (K)

        Returns:
            (mlptrain.sampling.umbrella.UmbrellaSampling):
        """
        us = cls(zeta_func=DummyCoordinate(), kappa=0.0, temp=temp)
        us.load(folder_name=folder_name)
        us._order_windows_by_zeta_ref()

        return us

    @classmethod
    def from_folders(cls,
                     *args: str,
                     temp: float) -> 'UmbrellaSampling':
        """
        Load a set of individual umbrella sampling simulations in to a single
        one

        -----------------------------------------------------------------------
        Arguments:
            *args: Names of folders

            temp: Temperature (K)

        Returns:
            (mlptrain.sampling.umbrella.UmbrellaSampling):
        """
        us = cls(zeta_func=DummyCoordinate(), kappa=0.0, temp=temp)

        for folder_name in args:
            us.load(folder_name=folder_name)

        us._order_windows_by_zeta_ref()
        return us

    def _order_windows_by_zeta_ref(self) -> None:
        """Sort the windows in this umbrella by ζ_ref"""
        self.windows = sorted(self.windows, key=lambda window: window.zeta_ref)
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

    @property
    def mean(self) -> float:
        """Mean of the Normal distribution, of which this is an approx."""
        return self.params[1]


def _plot_and_save_free_energy(free_energies,
                               zetas,
                               units='kcal mol-1') -> None:
    """
    Plots the free energy against the reaction coordinate and saves
    the corresponding values as a .txt file

    -----------------------------------------------------------------------
    Arguments:

        zetas: Values of the reaction coordinate
    """
    if units.lower() == 'ev':
        pass

    elif units.lower() == 'kcal mol-1':
        free_energies *= 23.060541945329334  # eV -> kcal mol-1

    elif units.lower() == 'kj mol-1':
        free_energies *= 96.48530749925793  # eV -> kJ mol-1

    else:
        raise ValueError(f'Unknown energy units: {units}')

    rel_free_energies = free_energies - min(free_energies)

    fig, ax = plt.subplots()
    ax.plot(zetas, rel_free_energies, color='k')

    with open(f'free_energy.txt', 'w') as outfile:
        for zeta, free_energy in zip(zetas, rel_free_energies):
            print(zeta, free_energy, file=outfile)

    ax.set_xlabel('Reaction coordinate / Å')
    ax.set_ylabel('ΔG / kcal mol$^{-1}$')

    fig.tight_layout()
    fig.savefig('free_energy.pdf')
    plt.close(fig)
    return None
