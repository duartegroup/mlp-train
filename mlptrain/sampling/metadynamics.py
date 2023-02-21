import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union, Tuple
from multiprocessing import Pool
from subprocess import Popen
from copy import deepcopy
from mlptrain.configurations import ConfigurationSet
from mlptrain.sampling.md import run_mlp_md
from mlptrain.sampling.plumed import PlumedBias
from mlptrain.utils import move_files, unique_dirname
from mlptrain.config import Config
from mlptrain.log import logger


class Metadynamics:
    """Metadynamics class for running biased molecular dynamics using
    metadynamics bias and analysing the results"""

    def __init__(self,
                 cvs: Union[Sequence['mlptrain._PlumedCV'],
                                     'mlptrain._PlumedCV'],
                 temp: Optional[float] = None):
        """
        Molecular dynamics using metadynamics bias. Used for calculating free
        energies (by using well-tempered metadynamics bias) and sampling
        configurations for active learning.

        -----------------------------------------------------------------------
        Arguments:

            cvs: Sequence of PLUMED collective variables
        """
        self.bias:     'mlptrain.PlumedBias' = PlumedBias(cvs)
        self.temp:     Optional[float] = temp                     # K

    @property
    def n_cvs(self) -> int:
        """Returns the number of collective variables used in metadynamics"""
        return len(self.bias.cvs)

    def run_metadynamics(self,
                         start_config: 'mlptrain.Configuration',
                         mlp: 'mlptrain.potentials._base.MLPotential',
                         temp: float,
                         interval: int,
                         dt: float,
                         pace: int,
                         width: Union[Sequence[float], float],
                         height: float,
                         biasfactor: float,
                         n_runs: int = 1,
                         n_walkers: int = 1,
                         save_sep: bool = False,
                         **kwargs) -> None:
        """
        Perform multiple well-tempered metadynamics runs in parallel, generate
        .xyz files containing trajectories of the runs, generate PLUMED files
        containing deposited gaussians and trajectories in terms of the CVs.

        -----------------------------------------------------------------------
        Arguments:

            start_config: Configuration from which the simulation is started

            mlp: Machine learnt potential

            temp (float): Temperature in K to initialise velocities and to run
                          NVT MD. Must be positive

            interval (int): Interval between saving the geometry

            dt (float): Time-step in fs

            pace (int): τ_G/dt, interval at which a new gaussian is placed

            width (float): σ, standard deviation (parameter describing the
                           width) of the placed gaussian

            height (float): ω, initial height of placed gaussians

            biasfactor (float): γ, describes how quickly gaussians shrink,
                                larger values make gaussians to be placed
                                less sensitive to the bias potential

            n_runs (int): Number of times to run metadynamics

            n_walkers (int): Number of walkers to use in each simulation

            save_sep (bool): If True saves trajectories of
                             each window separately

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """
        if temp <= 0:
            raise ValueError('Temperature must be positive and non-zero for '
                             'well-tempered metadynamics')

        start_metadynamics = time.perf_counter()

        self.temp = temp
        self.bias.set_metad_params(width=width,
                                   pace=pace,
                                   height=height,
                                   biasfactor=biasfactor)

        metad_processes, metad_trajs = [], []

        # TODO: Change if decide to use multiple walkers
        n_processes = min(Config.n_cores, n_runs)
        logger.info(f'Running {n_runs} independent Well-Tempered '
                    'Metadynamics simulation(s), '
                    f'{n_processes} simulation(s) are run parallel, '
                    f'{n_walkers} walker(s) per simulation')

        with Pool(processes=n_processes) as pool:

            for idx in range(n_runs):

                # Without copy kwargs is overwritten at every iteration
                kwargs_single = deepcopy(kwargs)
                kwargs_single['_idx'] = idx

                metad_process = pool.apply_async(func=self._run_single_metadynamics,
                                                 args=(start_config,
                                                       mlp,
                                                       temp,
                                                       interval,
                                                       dt,
                                                       self.bias),
                                                 kwds=kwargs_single)
                metad_processes.append(metad_process)

            for metad_process in metad_processes:
                metad_trajs.append(metad_process.get())

        if save_sep:
            metad_folder = unique_dirname('metad_trajectories')
            os.mkdir(metad_folder)

            for idx, metad_traj in enumerate(metad_trajs):
                metad_traj.save(filename=os.path.join(metad_folder,
                                                      f'metad_{idx}.xyz'))

        else:
            combined_traj = ConfigurationSet()
            for metad_traj in metad_trajs:
                combined_traj += metad_traj

            combined_traj.save(filename='combined_trajectory.xyz')

        move_files(['.dat'], 'plumed_files')
        move_files(['.log'], 'plumed_logs')

        finish_metadynamics = time.perf_counter()
        logger.info('Metadynamics done in '
                    f'{(finish_metadynamics - start_metadynamics) / 60:.1f} m')

        return None

    def _run_single_metadynamics(self, start_config, mlp, temp, interval, dt,
                                 bias, **kwargs):
        """Initiates a single well-tempered metadynamics run"""

        logger.info('Running Metadynamics simulation number '
                    f'{kwargs["_idx"]+1}')

        kwargs['n_cores'] = 1
        kwargs['_method'] = 'metadynamics'

        traj = run_mlp_md(configuration=start_config,
                          mlp=mlp,
                          temp=temp,
                          dt=dt,
                          interval=interval,
                          bias=bias,
                          **kwargs)

        return traj

    def compute_fes(self,
                    n_bins: int = 300,
                    cvs_bounds: Optional[Sequence] = None) -> np.ndarray:
        """
        Computes fes.dat files using generated HILLS.dat files from metadynamics
        simulation, using fes.dat files creates grids which contain collective
        variables and free energy surfaces, and saves the grids in .npy format
        which can be used to plot FES.

        -----------------------------------------------------------------------
        Arguments:

            n_bins (int): Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
                        e.g. [(0.8, 1.5), (80, 120)]
        """

        if not os.path.exists('plumed_files'):
            raise FileNotFoundError('Folder with PLUMED files was not found, '
                                    'make sure to run metadynamics before '
                                    'computing the FES')

        logger.info('Computing the free energy surface')

        os.chdir('plumed_files')

        self._compute_fes_files(n_bins, cvs_bounds)

        grid_shape = tuple([n_bins+1 for _ in range(self.n_cvs)])

        # Compute CV grids
        cv_grids = []
        for filename in os.listdir():
            if filename.startswith('fes'):

                for idx in range(self.n_cvs):
                    cv_vector = np.loadtxt(filename, usecols=idx)

                    cv_grid = np.reshape(cv_vector, grid_shape)
                    cv_grids.append(cv_grid)

                # All fes files would generate same grids -> can break
                break

        # Compute fes grids
        fes_grids = []
        for filename in os.listdir():
            if filename.startswith('fes'):

                fes_vector = np.loadtxt(filename,
                                        usecols=self.n_cvs)

                fes_grid = np.reshape(fes_vector, grid_shape)
                fes_grids.append(fes_grid)

        total_cv_grid = np.stack(cv_grids, axis=0)
        total_fes_grid = np.stack(fes_grids, axis=0)

        fes_raw = np.concatenate((total_cv_grid, total_fes_grid), axis=0)
        np.save('../fes_raw.npy', fes_raw)

        mean_fes_grid = np.mean(total_fes_grid, axis=0)
        std_fes_grid = np.std(total_fes_grid, axis=0)
        statistical_fes_grid = np.stack((mean_fes_grid, std_fes_grid), axis=0)

        fes = np.concatenate((total_cv_grid, statistical_fes_grid), axis=0)
        np.save('../fes.npy', fes)

        os.chdir('..')

        return fes

    def plot_fes(self,
                 fes: Union[np.ndarray, str]) -> None:
        """
        Plots FES (mean and standard deviation from multiple runs) for
        metadynamics simulations involving one or two collective variables

        -----------------------------------------------------------------------
        Arguments:

            fes (np.ndarray | str): Numpy grid of the free energy surface to be
                                    plotted
        """

        if self.n_cvs == 1:
            self._plot_1d_fes(fes)

        elif self.n_cvs == 2:
            self._plot_2d_fes(fes)

        else:
            raise NotImplementedError('Plotting FES is only available for one '
                                      'or two collective variables')

        return None

    def _compute_fes_files(self, n_bins, cvs_bounds) -> None:
        """Generate fes.dat files from HILLS.dat files"""

        if not any(filename.startswith('HILLS') for filename in os.listdir()):
            raise FileNotFoundError('No HILLS.dat files were found in '
                                    'plumed_files, make sure to run '
                                    'metadynamics before computing the FES')

        logger.info('Generating fes.dat files from HILLS.dat files')

        bin_param_seq = ','.join(str(n_bins) for _ in range(self.n_cvs))
        min_param_seq, max_param_seq = self._get_min_max_params(cvs_bounds)

        for filename in os.listdir():

            if filename.startswith('HILLS'):

                # HILLS_*.dat -> *
                name_specifier = filename.split('_')[1][:-4]

                with open(f'fes_{name_specifier}.log', 'w') as logfile:
                    compute_fes = Popen(['plumed', 'sum_hills',
                                         '--hills', filename,
                                         '--outfile', f'fes_{name_specifier}.dat',
                                         '--bin', bin_param_seq,
                                         '--min', min_param_seq,
                                         '--max', max_param_seq],
                                        stdout=logfile,
                                        stderr=logfile)
                    compute_fes.wait()

        move_files(['.log'], '../plumed_logs')

        return None

    def _get_min_max_params(self, cvs_bounds) -> Tuple:
        """Compute min and max parameters for generating fes.dat files from
        HILLS.dat files"""

        if cvs_bounds is None:
            logger.info('CVs bounds were not supplied, generating min and max '
                        'parameters automatically')

            min_params, max_params = [], []

            for cv in self.bias.cvs:
                min_values, max_values = [], []

                for filename in os.listdir():
                    if filename.startswith(f'colvar_{cv.name}'):
                        cv_values = np.loadtxt(filename, usecols=1)

                        min_values.append(np.min(cv_values))
                        max_values.append(np.max(cv_values))

                total_min = min(min_values)
                total_max = max(max_values)

                # TODO: Change based on sigma probably
                extension = 0.2 * (total_max - total_min)

                min_params.append(str(total_min - extension))
                max_params.append(str(total_max + extension))

            return ','.join(min_params), ','.join(max_params)

        cvs_bounds_checked = self._check_cv_bounds(cvs_bounds)

        min_params = [str(cv_bounds[0]) for cv_bounds in cvs_bounds_checked]
        max_params = [str(cv_bounds[1]) for cv_bounds in cvs_bounds_checked]

        return ','.join(min_params), ','.join(max_params)

    def _check_cv_bounds(self, cvs_bounds) -> Sequence:
        """doc"""

        cvs_bounds_checked = []

        if (not isinstance(cvs_bounds, tuple)
                and not isinstance(cvs_bounds, list)):

            raise TypeError('CVs_bounds must be a tuple or a list')

        elif len(cvs_bounds) == 0:

            raise TypeError('CVs_bounds cannot be an empty list '
                            'or an empty tuple')

        elif (all(isinstance(cv_bounds, tuple)
                  or isinstance(cv_bounds, list) for cv_bounds in cvs_bounds)):

            for cv_bounds in cvs_bounds:
                if len(cv_bounds) != 2 or not all(isinstance(cv_bound, float)
                                                  for cv_bound in cv_bounds):
                    raise ValueError('Supplied bounds must be sequences '
                                     'of two floats')

                cvs_bounds_checked.append(sorted(cv_bounds))

        elif all(isinstance(cv_bound, float) for cv_bound in cvs_bounds):
            if len(cvs_bounds) != 2:
                raise ValueError('Supplied bounds must be a sequence '
                                 'of two floats')

            cvs_bounds_checked.append(sorted(cvs_bounds))

        else:
            raise TypeError('CVs_bounds must be a sequence of sequences of '
                            'two floats, or a sequence of two floats')

        if len(cvs_bounds_checked) != self.n_cvs:
            raise ValueError('The number of supplied CV bounds does not agree '
                             'with the number of CVs used in metadynamics')

        return cvs_bounds_checked

    def _plot_1d_fes(self, fes) -> None:
        """Plots 1D mean free energy surface with standard error from multiple
        metadynamics runs"""

        cv = fes[0]
        mean_fes = fes[1]
        std_error = fes[2]

        lower_bound = mean_fes - 1/2 * std_error
        upper_bound = mean_fes + 1/2 * std_error

        fig, ax = plt.subplots()

        ax.plot(cv, mean_fes)
        ax.fill_between(cv, lower_bound, upper_bound, alpha=0.5)

        ax.set_xlabel('Reaction coordinate')
        ax.set_ylabel('ΔG')  # TODO: Change units from ASE to something else

        fig.tight_layout()

        fig.savefig('metad_free_energy.pdf')
        plt.close(fig)

        return None

    def _plot_2d_fes(self, fes) -> None:
        """Plots 2D mean free energy surface and standard error from multiple
        metadynamics runs"""

        cv1 = fes[0]
        cv2 = fes[1]
        mean_fes = fes[2]
        std_error = fes[3]

        fig, (ax_mean, ax_std_error) = plt.subplots(nrows=1, ncols=2,
                                                    figsize=(10, 5))

        mean_contourf = ax_mean.contourf(cv1, cv2, mean_fes, 10,
                                         cmap='turbo')
        ax_mean.contour = (cv1, cv2, mean_fes, 10)

        mean_cbar = fig.colorbar(mean_contourf, ax=ax_mean)
        mean_cbar.set_label(label=r'Free Energy $\Delta G$')

        std_error_contourf = ax_std_error.contourf(cv1, cv2, std_error, 10,
                                                   cmap='Blues')
        ax_std_error.contour = (cv1, cv2, mean_fes, 10)

        std_error_cbar = fig.colorbar(std_error_contourf, ax=ax_std_error)
        std_error_cbar.set_label(label=r'Standard Error $\sigma$')

        fig.tight_layout()

        fig.savefig('metad_free_energy.pdf')
        plt.close(fig)

        return None
