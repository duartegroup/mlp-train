import os
import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union, Tuple, List
from multiprocessing import Pool
from subprocess import Popen
from copy import deepcopy
from mlptrain.configurations import ConfigurationSet
from mlptrain.sampling.md import run_mlp_md
from mlptrain.sampling.plumed import PlumedBias, plot_cv, plot_trajectory
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

    def estimate_width(self,
                       configurations: Union['mlptrain.Configuration',
                                             'mlptrain.ConfigurationSet'],
                       mlp: 'mlptrain.potentials._base.MLPotential',
                       plot: bool = False,
                       **kwargs) -> List:
        """
        Estimates optimal widths (σ) to be used in metadynamics.

        -----------------------------------------------------------------------
        Arguments:

            configurations: A set of all known minima configurations that are
                            likely to be visited during metadynamics

            mlp: Machine learnt potential

            plot (bool): If True plots trajectories of collective variables as
                         a function of time

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units

        -------------------
        Returns:

            (List): List of optimal width values for each CV,
                    e.g. [0.02, 0.03] for two CVs
        """

        if not any(key in kwargs for key in ['fs', 'ps', 'ns']):
            kwargs['ps'] = 10

        temp = 300
        dt = 1
        interval = 10

        configuration_set = ConfigurationSet()
        configuration_set = configuration_set + configurations

        logger.info('Estimating optimal width (σ)')

        width_processes = []
        all_widths = []

        n_processes = min(Config.n_cores, len(configuration_set))

        # Spawn is likely to make it slower, but fork in combination
        # with plotting gives errors on MacOS > 10.13
        with mp.get_context('spawn').Pool(processes=n_processes) as pool:

            for idx, configuration in enumerate(configuration_set):

                kwargs_single = deepcopy(kwargs)
                kwargs_single['_idx'] = idx + 1

                width_process = pool.apply_async(func=self._get_width_for_single,
                                                 args=(configuration,
                                                       mlp,
                                                       temp,
                                                       dt,
                                                       interval,
                                                       self.bias,
                                                       plot),
                                                 kwds=kwargs_single)
                width_processes.append(width_process)

            for width_process in width_processes:
                all_widths.append(width_process.get())

        return [min(all_widths[:][idx]) for idx in range(self.n_cvs)]

    def _get_width_for_single(self, configuration, mlp, temp, dt, interval,
                              bias, plot, **kwargs) -> List:
        """Estimates optimal widths (σ) for a single configuration"""

        logger.info(f'Running MD simulation number {kwargs["_idx"]}')

        kwargs['n_cores'] = 1

        run_mlp_md(configuration=configuration,
                   mlp=mlp,
                   temp=temp,
                   dt=dt,
                   interval=interval,
                   bias=bias,
                   **kwargs)

        move_files(['.dat'], dst_folder='plumed_files/width_estimation')
        move_files(['.log'], dst_folder='plumed_logs/width_estimation')
        os.chdir('plumed_files/width_estimation')

        widths = []

        for cv in self.bias.cvs:
            colvar_filename = f'colvar_{cv.name}_{kwargs["_idx"]}.dat'

            cv_array = np.loadtxt(colvar_filename, usecols=1)

            width = np.std(cv_array)
            widths.append(width)

            if plot is True:
                plot_cv(filename=colvar_filename,
                        cv_units=cv.units,
                        label=str(kwargs["_idx"]))

        os.chdir('../..')

        return widths

    def run_metadynamics(self,
                         configuration: 'mlptrain.Configuration',
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

            configuration: Configuration from which the simulation is started

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

        start_metad = time.perf_counter()

        self.temp = temp
        self.bias._set_metad_params(width=width,
                                    pace=pace,
                                    height=height,
                                    biasfactor=biasfactor)

        metad_processes, metad_trajs = [], []

        # TODO: Change if decide to use multiple walkers
        n_processes = min(Config.n_cores, n_runs)
        logger.info(f'Running {n_runs} independent Well-Tempered '
                    'Metadynamics simulation(s), '
                    f'{n_processes} simulation(s) run in parallel, '
                    f'{n_walkers} walker(s) per simulation')

        with Pool(processes=n_processes) as pool:

            for idx in range(n_runs):

                # Without copy kwargs is overwritten at every iteration
                kwargs_single = deepcopy(kwargs)
                kwargs_single['_idx'] = idx + 1

                metad_process = pool.apply_async(func=self._run_single_metad,
                                                 args=(configuration,
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
                                                      f'metad_{idx+1}.xyz'))

        else:
            combined_traj = ConfigurationSet()
            for metad_traj in metad_trajs:
                combined_traj += metad_traj

            combined_traj.save(filename='combined_trajectory.xyz')

        move_files(['.dat'], dst_folder='plumed_files')
        move_files(['.log'], dst_folder='plumed_logs')

        finish_metad = time.perf_counter()
        logger.info('Metadynamics done in '
                    f'{(finish_metad - start_metad) / 60:.1f} m')

        return None

    def _run_single_metad(self, configuration, mlp, temp, interval, dt, bias,
                          **kwargs):
        """Initiates a single well-tempered metadynamics run"""

        logger.info(f'Running Metadynamics simulation number '
                    f'{kwargs["_idx"]}')

        kwargs['n_cores'] = 1
        kwargs['_method'] = 'metadynamics'

        traj = run_mlp_md(configuration=configuration,
                          mlp=mlp,
                          temp=temp,
                          dt=dt,
                          interval=interval,
                          bias=bias,
                          **kwargs)

        return traj

    def try_multiple_biasfactors(self,
                                 configuration: 'mlptrain.Configuration',
                                 mlp: 'mlptrain.potentials._base.MLPotential',
                                 temp: float,
                                 interval: int,
                                 dt: float,
                                 pace: int,
                                 width: Union[Sequence[float], float],
                                 height: float,
                                 biasfactors: Sequence[float],
                                 plotted_cvs: Optional = None,
                                 n_walkers: int = 1,
                                 **kwargs) -> None:
        """
        Executes multiple well-tempered metadynamics runs in parallel with a
        provided sequence of biasfactors and plots the resulting trajectories,
        useful for estimating the optimal biasfactor value.

        -----------------------------------------------------------------------
        Arguments:

            configuration: Configuration from which the simulation is started

            mlp: Machine learnt potential

            temp (float): Temperature in K to initialise velocities and to run
                          NVT MD. Must be positive

            interval (int): Interval between saving the geometry

            dt (float): Time-step in fs

            pace (int): τ_G/dt, interval at which a new gaussian is placed

            width (float): σ, standard deviation (parameter describing the
                           width) of the placed gaussian

            height (float): ω, initial height of placed gaussians

            biasfactors: Sequence of γ, describes how quickly gaussians shrink,
                         larger values make gaussians to be placed less
                         sensitive to the bias potential

            plotted_cvs: Sequence of one or two PlumedCV objects which are
                         going to be plotted, must be a subset for CVs used to
                         define Metadynamics

            n_walkers (int): Number of walkers to use in each simulation

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        logger.info(f'Trying {len(biasfactors)} different biasfactors')

        # Dummy bias which stores cvs, useful for checking cvs input
        if plotted_cvs is not None:
            cvs_holder = PlumedBias(plotted_cvs)

        else:
            cvs_holder = self.bias

        if len(cvs_holder.cvs) > 2:
            raise NotImplementedError('Plotting using more than two CVs is not '
                                      'implemented')

        if not all(cv in self.bias.cvs for cv in cvs_holder.cvs):
            raise ValueError('At least one of the supplied CVs are not within '
                             'the set of CVs used to define Metadynamics')

        self.bias._set_metad_params(width=width,
                                    pace=pace,
                                    height=height)

        n_processes = min(Config.n_cores, len(biasfactors))
        logger.info('Running Well-Tempered Metadynamics simulations '
                    f'with {len(biasfactors)} different biasfactors, '
                    f'{n_processes} simulation(s) run in parallel, '
                    f'{n_walkers} walker(s) per simulation')

        with mp.get_context('spawn').Pool(processes=n_processes) as pool:

            for idx, biasfactor in enumerate(biasfactors):

                bias = deepcopy(self.bias)
                bias.biasfactor = biasfactor

                kwargs_single = deepcopy(kwargs)
                kwargs_single['_idx'] = idx + 1

                pool.apply_async(func=self._try_single_biasfactor,
                                 args=(configuration,
                                       mlp,
                                       temp,
                                       interval,
                                       dt,
                                       bias,
                                       cvs_holder.cvs),
                                 kwds=kwargs_single)

            pool.close()
            pool.join()

        return None

    def _try_single_biasfactor(self, configuration, mlp, temp, interval, dt,
                               bias, plotted_cvs, **kwargs):
        """Executes a single well-tempered metadynamics run and plots the
        resulting trajectory"""

        self._run_single_metad(configuration=configuration,
                               mlp=mlp,
                               temp=temp,
                               interval=interval,
                               dt=dt,
                               bias=bias,
                               **kwargs)

        move_files(['.dat'], dst_folder='plumed_files/multiple_biasfactors')
        move_files(['.log'], dst_folder='plumed_logs/multiple_biasfactors')
        os.chdir('plumed_files/multiple_biasfactors')

        filenames = [f'colvar_{cv.name}_{kwargs["_idx"]}.dat'
                     for cv in plotted_cvs]

        for filename, cv in zip(filenames, plotted_cvs):
            plot_cv(filename=filename,
                    cv_units=cv.units,
                    label=str(kwargs['_idx']))

        if len(plotted_cvs) == 2:
            plot_trajectory(filenames=filenames,
                            cvs_units=[cv.units for cv in plotted_cvs],
                            label=str(kwargs['_idx']))

        os.chdir('../..')

        return None

    def compute_fes(self,
                    n_bins: int = 300,
                    cvs_bounds: Optional[Sequence] = None) -> np.ndarray:
        """
        Computes fes.dat files using generated HILLS.dat files from metadynamics
        simulation, using fes.dat files creates grids which contain collective
        variables and free energy surfaces (in eV), and saves the grids in .npy
        format which can be used to plot the FES (other way to generate plots
        is to open fes.dat with gnuplot).

        -----------------------------------------------------------------------
        Arguments:

            n_bins (int): Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
                        e.g. [(0.8, 1.5), (80, 120)]
        """

        if not os.path.exists('plumed_files'):
            raise FileNotFoundError('Folder with PLUMED files not found, '
                                    'make sure to run metadynamics before '
                                    'computing the FES')

        os.chdir('plumed_files')

        self._compute_fes_files(n_bins, cvs_bounds)

        logger.info('Computing the free energy surface')

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

        move_files(['.log'], dst_folder='../plumed_logs')

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

        if isinstance(cvs_bounds, list) or isinstance(cvs_bounds, tuple):

            if len(cvs_bounds) == 0:
                raise TypeError('CVs bounds cannot be an empty list or '
                                'an empty tuple')

            elif all(isinstance(cv_bounds, list) or isinstance(cv_bounds, tuple)
                     for cv_bounds in cvs_bounds):
                _cvs_bounds = cvs_bounds

            elif all(isinstance(cv_bound, float) or isinstance(cv_bound, int)
                     for cv_bound in cvs_bounds):
                _cvs_bounds = [cvs_bounds]

            else:
                raise TypeError('CVs bounds are in incorrect format')

        else:
            raise TypeError('CVs bounds are in incorrect format')

        if len(_cvs_bounds) != self.n_cvs:
            raise ValueError('The number of supplied CVs bounds is not equal '
                             'to the number of CVs used in metadynamics')

        return _cvs_bounds

    def plot_fes(self,
                 fes: Union[np.ndarray, str],
                 units: str = 'kcal mol-1') -> None:
        """
        Plots FES (mean and standard deviation from multiple runs) for
        metadynamics simulations involving one or two collective variables

        -----------------------------------------------------------------------
        Arguments:

            fes (np.ndarray | str): Numpy grid of the free energy surface to be
                                    plotted
        """

        if self.n_cvs == 1:
            self._plot_1d_fes(fes, units)

        elif self.n_cvs == 2:
            self._plot_2d_fes(fes, units)

        else:
            raise NotImplementedError('Plotting FES is only available for one '
                                      'and two collective variables')

        return None

    def _plot_1d_fes(self, fes, units) -> None:
        """Plots 1D mean free energy surface with standard error from multiple
        metadynamics runs"""

        logger.info('Plotting 1D FES')

        cv = fes[0]
        mean_fes = fes[1]
        std_error = fes[2]

        if units.lower() == 'ev':
            pass

        elif units.lower() == 'kcal mol-1':
            mean_fes *= 23.060541945329334  # eV -> kcal mol-1
            std_error *= 23.060541945329334

        elif units.lower() == 'kj mol-1':
            mean_fes *= 96.48530749925793  # eV -> kJ mol-1
            std_error *= 96.48530749925793

        else:
            raise ValueError(f'Unknown energy units: {units}')

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

    def _plot_2d_fes(self, fes, units) -> None:
        """Plots 2D mean free energy surface and standard error from multiple
        metadynamics runs"""

        logger.info('Plotting 2D FES')

        cv1 = fes[0]
        cv2 = fes[1]
        mean_fes = fes[2]
        std_error = fes[3]

        if units.lower() == 'ev':
            pass

        elif units.lower() == 'kcal mol-1':
            mean_fes *= 23.060541945329334  # eV -> kcal mol-1
            std_error *= 23.060541945329334

        elif units.lower() == 'kj mol-1':
            mean_fes *= 96.48530749925793  # eV -> kJ mol-1
            std_error *= 96.48530749925793

        else:
            raise ValueError(f'Unknown energy units: {units}')

        fig, (ax_mean, ax_std_error) = plt.subplots(nrows=1, ncols=2,
                                                    figsize=(10, 5))

        mean_contourf = ax_mean.contourf(cv1, cv2, mean_fes, 20,
                                         cmap='turbo')
        ax_mean.contour = (cv1, cv2, mean_fes, 20)

        mean_cbar = fig.colorbar(mean_contourf, ax=ax_mean)
        mean_cbar.set_label(label=r'Free Energy $\Delta G$')

        std_error_contourf = ax_std_error.contourf(cv1, cv2, std_error, 20,
                                                   cmap='Blues')
        ax_std_error.contour = (cv1, cv2, mean_fes, 20)

        std_error_cbar = fig.colorbar(std_error_contourf, ax=ax_std_error)
        std_error_cbar.set_label(label=r'Standard Error $\sigma$')

        fig.tight_layout()
        fig.savefig('metad_free_energy.pdf')
        plt.close(fig)

        return None
