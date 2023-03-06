import os
import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union, Tuple, List
from multiprocessing import Pool
from subprocess import Popen
from copy import deepcopy
from ase import units as ase_units
from mlptrain.configurations import ConfigurationSet
from mlptrain.sampling.md import run_mlp_md
from mlptrain.sampling.plumed import PlumedBias, plot_cv, plot_trajectory
from mlptrain.config import Config
from mlptrain.log import logger
from mlptrain.utils import (
    unique_name,
    move_files,
    convert_ase_time,
    convert_ase_energy,
    convert_exponents,
)


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
        """Number of collective variables used in metadynamics"""
        return len(self.bias.cvs)

    @property
    def kbt(self):
        """Value of k_B*T in ASE units"""
        return ase_units.kB * self.temp

    def estimate_width(self,
                       configurations: Union['mlptrain.Configuration',
                                             'mlptrain.ConfigurationSet'],
                       mlp:            'mlptrain.potentials._base.MLPotential',
                       temp:           float = 300,
                       interval:       int = 10,
                       dt:             float = 1,
                       plot:           bool = False,
                       **kwargs
                       ) -> List:
        """
        Estimates optimal widths (σ) to be used in metadynamics.

        -----------------------------------------------------------------------
        Arguments:

            configurations: A set of all known minima configurations that are
                            likely to be visited during metadynamics

            mlp: Machine learnt potential

            temp: (float) Temperature in K to initialise velocities and to run
                  NVT MD. Must be positive

            interval: (int) Interval between saving the geometry

            dt: (float) Time-step in fs

            plot: (bool) If True plots trajectories of collective variables as
                  a function of time

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units

        -------------------
        Returns:

            (List): List of optimal width values for each CV,
                    e.g. [0.02, 0.03] for two CVs
        """

        start = time.perf_counter()

        if not any(key in kwargs for key in ['fs', 'ps', 'ns']):
            kwargs['ps'] = 10

        configuration_set = ConfigurationSet()
        configuration_set = configuration_set + configurations

        logger.info('Estimating optimal width (σ)')

        width_processes, all_widths = [], []

        n_processes = min(Config.n_cores, len(configuration_set))

        # Spawn is likely to make it slower, but fork in combination
        # with plotting is likely to give errors on MacOS > 10.13
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
                all_widths = np.array(all_widths)

        move_files([r'colvar_\w+_\d+\.dat'],
                   dst_folder='plumed_files/width_estimation',
                   regex=True)

        move_files([r'\w+_config\d+\.pdf'],
                   dst_folder='width_estimation',
                   regex=True)

        finish = time.perf_counter()
        logger.info(f'Width estimation done in {(finish - start) / 60:.1f} m')

        return list(np.min(all_widths, axis=0))

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
                   kept_substrings=['.dat'],
                   **kwargs)

        widths = []

        for cv in self.bias.cvs:
            colvar_filename = f'colvar_{cv.name}_{kwargs["_idx"]}.dat'

            cv_array = np.loadtxt(colvar_filename, usecols=1)

            width = np.std(cv_array)
            widths.append(width)

            if plot is True:
                plot_cv(filename=colvar_filename,
                        cv_units=cv.units,
                        label=f'config{kwargs["_idx"]}')

        return widths

    def run_metadynamics(self,
                         configuration: 'mlptrain.Configuration',
                         mlp:           'mlptrain.potentials._base.MLPotential',
                         temp:          float,
                         interval:      int,
                         dt:            float,
                         pace:          int = 500,
                         height:        Optional[float] = None,
                         width:         Optional = None,
                         biasfactor:    Optional[float] = None,
                         n_runs:        int = 1,
                         save_sep:      bool = True,
                         restart:       bool = False,
                         **kwargs
                         ) -> None:
        """
        Perform multiple metadynamics runs in parallel, generate .xyz and .traj
        files containing trajectories of the runs, generate PLUMED files
        containing deposited gaussians and trajectories in terms of the CVs.

        -----------------------------------------------------------------------
        Arguments:

            configuration: Configuration from which the simulation is started

            mlp: Machine learnt potential

            temp: (float) Temperature in K to initialise velocities and to run
                  NVT MD. Must be positive

            interval: (int) Interval between saving the geometry

            dt: (float) Time-step in fs

            pace: (int) τ_G/dt, interval at which a new gaussian is placed

            height: (float) ω, initial height of placed gaussians

            width: (List[float] | float | None) σ, standard deviation
                   (parameter describing the width) of placed gaussians,
                   if not supplied it is estimated automatically

            biasfactor: (float) γ, describes how quickly gaussians shrink,
                                larger values make gaussians to be placed
                                less sensitive to the bias potential

            n_runs: (int) Number of times to run metadynamics

            save_sep: (bool) If True saves trajectories of
                             each simulation separately

            restart: (bool) If True restarts the most recent metadynamics
                            simulation
        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        start_metad = time.perf_counter()

        self.temp = temp
        if height is None:
            if temp > 0:
                logger.info('Height was not supplied, '
                            'setting height to 0.5*k_B*T')

                height = 0.5 * self.kbt

            else:
                raise ValueError('Height was not supplied')

        if biasfactor is not None and temp <= 0:
            raise ValueError('Temperature must be positive and non-zero for '
                             'well-tempered metadynamics')

        if restart:
            if width is None:
                raise ValueError('Make sure to use exactly the same width as '
                                 'in the previous simulation')

            if not os.path.exists('plumed_files/metadynamics'):
                raise FileNotFoundError('Metadynamics folder not found, make '
                                        'sure to run metadynamics before '
                                        'trying to restart')

            n_colvar_files = len([filename for filename in
                                  os.listdir('plumed_files/metadynamics')
                                  if 'colvar' in filename])

            if n_colvar_files != n_runs:
                raise NotImplementedError('Restart is implemented only if the '
                                          'number of runs matches the number '
                                          'of runs in the previous simulation')

            metad_path = os.path.join(os.getcwd(), 'plumed_files/metadynamics')
            traj_path = os.path.join(os.getcwd(), 'trajectories')

            for filename in os.listdir(metad_path):
                if filename.startswith('fes'):
                    os.remove(os.path.join(metad_path, filename))

            move_files(['.dat'],
                       dst_folder=os.getcwd(),
                       src_folder=metad_path,
                       unique=False)

            move_files(['.traj'],
                       dst_folder=os.getcwd(),
                       src_folder=traj_path,
                       unique=False)

            kept_substrings = None

        else:
            if width is None:
                logger.info('Width parameters were not supplied to the '
                            'metadynamics simulation, estimating widths '
                            'automatically')

                width = self.estimate_width(configurations=configuration,
                                            mlp=mlp)

            kept_substrings = ['.traj', '.dat']

        self.bias._set_metad_params(width=width,
                                    pace=pace,
                                    height=height,
                                    biasfactor=biasfactor)

        metad_processes, metad_trajs = [], []

        n_processes = min(Config.n_cores, n_runs)
        logger.info(f'Running {n_runs} independent Metadynamics simulation(s), '
                    f'{n_processes} simulation(s) run in parallel, '
                    f'1 walker per simulation')

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
                                                       self.bias,
                                                       kept_substrings,
                                                       restart),
                                                 kwds=kwargs_single)
                metad_processes.append(metad_process)

            for metad_process in metad_processes:
                metad_trajs.append(metad_process.get())

        if save_sep:
            for idx, metad_traj in enumerate(metad_trajs):
                metad_traj.save(filename=f'metad_{idx+1}.xyz')

            move_files([r'metad_\d+\.xyz'],
                       dst_folder='trajectories',
                       regex=True,
                       unique=True if not restart else False)

        else:
            combined_traj = ConfigurationSet()
            for metad_traj in metad_trajs:
                combined_traj += metad_traj

            combined_traj.save(filename='combined_trajectory.xyz')

            move_files(['combined_trajectory.xyz'],
                       dst_folder='trajectories',
                       unique=True if not restart else False)

        move_files([r'trajectory_\d+\.traj'],
                   dst_folder='trajectories',
                   regex=True,
                   unique=False)

        move_files(['.dat'],
                   dst_folder='plumed_files/metadynamics',
                   unique=True if not restart else False)

        if restart:
            self._remove_duplicate_frames()

        finish_metad = time.perf_counter()
        logger.info('Metadynamics done in '
                    f'{(finish_metad - start_metad) / 60:.1f} m')

        return None

    def _run_single_metad(self, configuration, mlp, temp, interval, dt, bias,
                          kept_substrings=None, restart=False, **kwargs):
        """Initiates a single metadynamics run"""

        if restart:
            logger.info(f'Restarting Metadynamics simulation number '
                        f'{kwargs["_idx"]}')

            restart_files = []

            for cv in self.bias.cvs:
                restart_files.append(f'colvar_{cv.name}_{kwargs["_idx"]}.dat')

            restart_files.append(f'HILLS_{kwargs["_idx"]}.dat')
            restart_files.append(f'trajectory_{kwargs["_idx"]}.traj')

        else:
            if bias.biasfactor is None:
                logger.info('Running Metadynamics simulation '
                            f'number {kwargs["_idx"]}')

            else:
                logger.info('Running Well-tempered Metadynamics simulation '
                            f'number {kwargs["_idx"]} '
                            f'with a biasfactor {bias.biasfactor}')

            restart_files = None

        kwargs['n_cores'] = 1
        kwargs['_method'] = 'metadynamics'

        traj = run_mlp_md(configuration=configuration,
                          mlp=mlp,
                          temp=temp,
                          dt=dt,
                          interval=interval,
                          bias=bias,
                          kept_substrings=kept_substrings,
                          restart_files=restart_files,
                          **kwargs)

        return traj

    @staticmethod
    def _remove_duplicate_frames() -> None:
        """Removes a duplicate frame from colvar files after restarting
        metadynamics"""

        os.chdir('plumed_files/metadynamics')

        for filename in os.listdir():
            if filename.startswith('colvar'):

                with open(filename, 'r') as f:
                    lines = f.readlines()

                duplicate_index = None
                for i, line in enumerate(lines):
                    if line.startswith('#!') and i != 0:

                        # First frame before the redundant header is a duplicate
                        duplicate_index = i - 1
                        break

                if duplicate_index is None:
                    raise TypeError('Duplicate frame was not found')

                lines.pop(duplicate_index)

                with open(filename, 'w') as f:
                    for line in lines:
                        f.write(line)

        os.chdir('../..')

        return None

    def try_multiple_biasfactors(self,
                                 configuration: 'mlptrain.Configuration',
                                 mlp:   'mlptrain.potentials._base.MLPotential',
                                 temp:          float,
                                 interval:      int,
                                 dt:            float,
                                 biasfactors:   Sequence[float],
                                 pace:          int = 500,
                                 height:        Optional[float] = None,
                                 width:         Optional = None,
                                 plotted_cvs:   Optional = None,
                                 **kwargs
                                 ) -> None:
        """
        Executes multiple well-tempered metadynamics runs in parallel with a
        provided sequence of biasfactors and plots the resulting trajectories,
        useful for estimating the optimal biasfactor value.

        -----------------------------------------------------------------------
        Arguments:

            configuration: Configuration from which the simulation is started

            mlp: Machine learnt potential

            temp: (float) Temperature in K to initialise velocities and to run
                          NVT MD. Must be positive

            interval: (int) Interval between saving the geometry

            dt: (float) Time-step in fs

            biasfactors: Sequence of γ, describes how quickly gaussians shrink,
                         larger values make gaussians to be placed less
                         sensitive to the bias potential

            pace: (int) τ_G/dt, interval at which a new gaussian is placed

            height: (float) ω, initial height of placed gaussians

            width: (List[float] | float | None) σ, standard deviation
                   (parameter describing the width) of placed gaussians,
                   if not supplied it is estimated automatically

            plotted_cvs: Sequence of one or two PlumedCV objects which are
                         going to be plotted, must be a subset for CVs used to
                         define the Metadynamics object

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units
        """

        logger.info(f'Trying {len(biasfactors)} different biasfactors')
        start = time.perf_counter()

        if temp <= 0:
            raise ValueError('Temperature must be positive and non-zero for '
                             'well-tempered metadynamics')

        self.temp = temp
        if height is None:
            logger.info('Height was not supplied, setting height to 0.5*k_B*T')

            height = 0.5 * self.kbt

        if width is None:
            logger.info('Width parameters were not supplied to the multiple '
                        'biasfactor simulation, estimating widths '
                        'automatically')

            width = self.estimate_width(configurations=configuration,
                                        mlp=mlp)

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
                             'the set of CVs used to define the Metadynamics '
                             'object')

        self.bias._set_metad_params(width=width,
                                    pace=pace,
                                    height=height)

        n_processes = min(Config.n_cores, len(biasfactors))
        logger.info('Running Well-Tempered Metadynamics simulations '
                    f'with {len(biasfactors)} different biasfactors, '
                    f'{n_processes} simulation(s) run in parallel, '
                    f'1 walker per simulation')

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

        move_files([r'colvar_\w+_\d+\.dat'],
                   dst_folder='plumed_files/multiple_biasfactors',
                   regex=True)

        move_files([r'\w+_biasf\d+\.pdf'],
                   dst_folder='multiple_biasfactors',
                   regex=True)

        finish = time.perf_counter()
        logger.info('Simulations with multiple biasfactors done in '
                    f'{(finish - start) / 60:.1f} m')

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
                               kept_substrings=['.dat'],
                               **kwargs)

        filenames = [f'colvar_{cv.name}_{kwargs["_idx"]}.dat'
                     for cv in plotted_cvs]

        for filename, cv in zip(filenames, plotted_cvs):
            plot_cv(filename=filename,
                    cv_units=cv.units,
                    label=f'biasf{bias.biasfactor}')

        if len(plotted_cvs) == 2:
            plot_trajectory(filenames=filenames,
                            cvs_units=[cv.units for cv in plotted_cvs],
                            label=f'biasf{bias.biasfactor}')

        return None

    def block_analysis(self) -> None:
        """
        Performs block analysis on the most recent metadynamics run.

        -----------------------------------------------------------------------
        Arguments:

        """

        return None

    def plot_fes(self,
                 energy_units:          str = 'kcal mol-1',
                 n_bins:                int = 300,
                 cvs_bounds:            Optional[Sequence] = None,
                 fes_npy:               Optional[str] = None,
                 block_analysis_error:  Optional[str] = None,
                 ) -> None:
        """
        Plots the free energy surface using a supplied .npy file. If the file
        is not given computes the file using HILLS.dat files from the most
        recent metadynamics run.

        -----------------------------------------------------------------------
        Arguments:

            energy_units: (str) Energy units to be used in plotting

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
                        e.g. [(0.8, 1.5), (80, 120)]

            fes_npy: (str) File name of the .npy file used for plotting

            block_analysis_error: (str) File name containing a standard error
                                        grid from block analysis
        """

        if fes_npy is None:
            if not os.path.exists('fes_raw.npy'):
                self.compute_fes(energy_units, n_bins, cvs_bounds)

            logger.info('Using fes_raw.npy for plotting')
            fes_npy = 'fes_raw.npy'

        if self.n_cvs == 1:
            self._plot_1d_fes(fes_npy, energy_units, block_analysis_error)

        elif self.n_cvs == 2:
            self._plot_2d_fes(fes_npy, energy_units, block_analysis_error)

        else:
            raise NotImplementedError('Plotting FES is available only for one '
                                      'and two collective variables')

        return None

    def compute_fes(self,
                    energy_units:         str = 'kcal mol-1',
                    n_bins:               int = 300,
                    cvs_bounds:           Optional[Sequence] = None,
                    ) -> None:
        """
        Generates fes.dat files from HILLS.dat files and computes a total grid
        containing collective variable grids and free energy surface grids,
        which is saved in the current directory as .npy file.

        -----------------------------------------------------------------------
        Arguments:

            energy_units: (str) Energy units to be used in plotting

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
                        e.g. [(0.8, 1.5), (80, 120)]
        """

        os.chdir('plumed_files/metadynamics')

        self._compute_fes_files(n_bins, cvs_bounds)
        cv_grids, fes_grids = self._fes_files_to_grids(energy_units, n_bins)

        os.chdir('../..')

        fes_raw = np.concatenate((cv_grids, fes_grids), axis=0)
        np.save('fes_raw.npy', fes_raw)

        return None

    @staticmethod
    def _plot_1d_fes(fes_npy:               str,
                     energy_units:          str = 'kcal mol-1',
                     block_analysis_error:  Optional[str] = None
                     ) -> None:
        """Plots 1D mean free energy surface with standard error"""

        logger.info('Plotting 1D FES')

        fes = np.load(fes_npy)

        cv = fes[0]
        fes_grids = fes[1:]

        mean_fes = np.mean(fes_grids, axis=0)

        if block_analysis_error is None:
            std_error = np.std(fes_grids, axis=0)

        else:
            std_error = 1 / np.sqrt(len(fes_grids)) * block_analysis_error

        lower_bound = mean_fes - 1/2 * std_error
        upper_bound = mean_fes + 1/2 * std_error

        fig, ax = plt.subplots()

        ax.plot(cv, mean_fes)
        ax.fill_between(cv, lower_bound, upper_bound, alpha=0.5)

        ax.set_xlabel('Reaction coordinate')
        ax.set_ylabel(r'$\Delta G$ / '
                      f'{convert_exponents(energy_units)}')

        fig.tight_layout()

        figname = 'metad_free_energy.pdf'
        if os.path.exists(figname):
            os.rename(figname, unique_name(figname))

        fig.savefig(figname)
        plt.close(fig)

        return None

    @staticmethod
    def _plot_2d_fes(fes_npy:               str,
                     energy_units:          str = 'kcal mol-1',
                     block_analysis_error:  Optional[str] = None
                     ) -> None:
        """Plots 2D mean free energy surface with standard error"""

        logger.info('Plotting 2D FES')

        fes = np.load(fes_npy)

        cv1 = fes[0]
        cv2 = fes[1]
        fes_grids = fes[2:]

        mean_fes = np.mean(fes_grids, axis=0)

        if block_analysis_error is None:
            std_error = np.std(fes_grids, axis=0)

        else:
            std_error = 1 / np.sqrt(len(fes_grids)) * block_analysis_error

        fig, (ax_mean, ax_std_error) = plt.subplots(nrows=1, ncols=2,
                                                    figsize=(10, 5))

        mean_contourf = ax_mean.contourf(cv1, cv2, mean_fes,
                                         range(0, 70),
                                         cmap='jet')
        ax_mean.contour = (cv1, cv2, mean_fes, range(0, 70, 5))

        mean_cbar = fig.colorbar(mean_contourf, ax=ax_mean)
        mean_cbar.set_label(label=r'$\Delta G$ / '
                                  f'{convert_exponents(energy_units)}')

        std_error_contourf = ax_std_error.contourf(cv1, cv2, std_error,
                                                   range(0, 70),
                                                   cmap='Blues')
        ax_std_error.contour = (cv1, cv2, mean_fes, range(0, 70, 5))

        std_error_cbar = fig.colorbar(std_error_contourf, ax=ax_std_error)
        std_error_cbar.set_label(label=r'$\sigma$ / '
                                       f'{convert_exponents(energy_units)}')

        fig.tight_layout()

        figname = 'metad_free_energy.pdf'
        if os.path.exists(figname):
            os.rename(figname, unique_name(figname))

        fig.savefig(figname)
        plt.close(fig)

        return None

    def plot_fes_convergence(self,
                             stride:       int,
                             n_surfaces:   int = 5,
                             time_units:   str = 'ps',
                             energy_units: str = 'kcal mol-1',
                             n_bins:       int = 300,
                             cvs_bounds:   Optional[Sequence] = None,
                             idx:          int = 1
                             ) -> None:
        """
        Computes multiple fes.dat files from a HILLS_idx.dat file by summing
        the deposited gaussians using a stride. Uses the computed files to plot
        multiple FESs as a function of simulation time.

        -----------------------------------------------------------------------
        Arguments:

            stride: (int) Interval that specifies the number new of gaussians
                          that is put into subsequent fes_idx_*.dat file

            n_surfaces: (int) Number of surfaces to be plotted (counting from
                              the last computed surface), must not exceed the
                              number of computed surfaces

            time_units: (str) Time units to be used in plotting

            energy_units: (str) Energy units to be used in plotting

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
                        e.g. [(0.8, 1.5), (80, 120)]

            idx: (int) Integer which specifies which metadynamics run to use
                       for plotting the FES convergence
        """

        os.chdir('plumed_files/metadynamics')

        deposit_time = np.loadtxt(f'HILLS_{idx}.dat', usecols=0)

        fes_time = [deposit_time[i]
                    for i in range(stride - 1, len(deposit_time), stride)]

        remove_duplicate = fes_time[-1] == deposit_time[-1]

        fes_time = convert_ase_time(np.array(fes_time), time_units)
        fes_time = np.round(fes_time, decimals=1)

        # sum_hills generates surfaces with the stride,
        # but it also always computes the final FES
        self._compute_fes_files(n_bins=n_bins,
                                cvs_bounds=cvs_bounds,
                                stride=stride,
                                idx=idx)

        move_files([fr'fes_{idx}_\d+\.dat'],
                   dst_folder='../fes_convergence',
                   regex=True)
        os.chdir('../fes_convergence')

        # Remove the final FES if it has already been computed with the stride
        if remove_duplicate:
            os.remove(f'fes_{idx}_{len(fes_time)}.dat')

        cv_grids, fes_grids = self._fes_files_to_grids(energy_units,
                                                       n_bins,
                                                       relative=True)

        fes_diff_grids = np.diff(fes_grids, axis=0)
        rms_diffs = [np.sqrt(np.mean(grid * grid)) for grid in fes_diff_grids]

        fig, ax = plt.subplots()
        ax.plot(fes_time[:-1], rms_diffs)

        ax.set_xlabel(f'Time / {time_units}')
        ax.set_ylabel(r'$\left\langle\Delta\Delta G^{2} \right\rangle^{1/2}$ / '
                      f'{convert_exponents(energy_units)}')

        fig.tight_layout()

        fig.savefig('fes_convergence_diff.pdf')
        plt.close(fig)

        if self.n_cvs == 1:
            plotted_cv = self.bias.cvs[0]

            if n_surfaces > len(fes_grids):
                raise ValueError('The number of surfaces requested to plot is '
                                 'larger than the number of computed surfaces')

            fig, ax = plt.subplots()
            for i in range(len(fes_grids) - n_surfaces, len(fes_grids)):
                ax.plot(cv_grids[0], fes_grids[i],
                        label=f'{fes_time[i]} {time_units}')

            ax.legend()

            if plotted_cv.units is not None:
                ax.set_xlabel(f'{plotted_cv.name} / {plotted_cv.units}')

            else:
                ax.set_xlabel(f'{plotted_cv.name}')

            ax.set_ylabel(f'ΔG / {convert_exponents(energy_units)}')

            fig.tight_layout()

            fig.savefig('fes_convergence.pdf')
            plt.close(fig)

        os.chdir('../..')

        move_files(['fes_convergence.pdf', 'fes_convergence_diff.pdf'],
                   dst_folder='fes_convergence',
                   src_folder='plumed_files/fes_convergence')

        return None

    def _compute_fes_files(self,
                           n_bins:      int,
                           cvs_bounds:  Optional[Sequence] = None,
                           stride:      Optional[int] = None,
                           idx:         Optional[int] = None
                           ) -> None:
        """
        Generates fes.dat files from a HILLS.dat file.

        -----------------------------------------------------------------------
        Arguments:

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,

            stride: (int) Interval that specifies the number new of gaussians
                          that is put into subsequent fes_idx_*.dat file

            idx: (int) Integer which specifies which metadynamics run to use
                       for plotting the FES convergence
        """

        if not any(filename.startswith('HILLS') for filename in os.listdir()):
            raise FileNotFoundError('No HILLS.dat files were found in '
                                    'plumed_files, make sure to run '
                                    'metadynamics before computing the FES')

        logger.info('Generating fes.dat files from HILLS.dat files')

        bin_param_seq = ','.join(str(n_bins) for _ in range(self.n_cvs))
        min_param_seq, max_param_seq = self._get_min_max_params(cvs_bounds)

        hills_filename_start = 'HILLS' if idx is None else f'HILLS_{idx}'

        for filename in os.listdir():

            if filename.startswith(hills_filename_start):

                # HILLS_*.dat -> *
                index = filename[:-4].split('_')[-1]

                if stride is None:
                    fes_filename = f'fes_{index}.dat'
                    stride_setup = ''

                else:
                    fes_filename = f'fes_{index}_'
                    stride_setup = ['--stride', f'{stride}']

                compute_fes = Popen(['plumed', 'sum_hills',
                                     '--hills', filename,
                                     '--outfile', fes_filename,
                                     '--bin', bin_param_seq,
                                     '--min', min_param_seq,
                                     '--max', max_param_seq,
                                     *stride_setup])
                compute_fes.wait()

        return None

    def _fes_files_to_grids(self,
                            energy_units: str,
                            n_bins:       int,
                            relative:     bool = False
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses fes.dat files in a current directory to compute a grid containing
        collective variables and a grid containing free energy surfaces.

        -----------------------------------------------------------------------
        Arguments:

            energy_units: (str) Energy units to be used in plotting

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

        Returns:

            (Tuple): Two grids containing CVs and FESs
        """

        grid_shape = tuple([n_bins+1 for _ in range(self.n_cvs)])

        # Sort names to retain ordering in the final grid
        unordered_fes_files = [name for name in os.listdir() if 'fes' in name]

        # 'fes_1_12.dat' -> int(112)
        def _get_combined_index(name):
            name_without_extension = name.split('.')[0]
            indexes = name_without_extension.split('_')[1:]
            combined_index = ''.join(indexes)

            return int(combined_index)

        fes_files = sorted(unordered_fes_files, key=_get_combined_index)

        # Compute CV grids
        cv_grids = []
        for filename in fes_files:

            for idx in range(self.n_cvs):
                cv_vector = np.loadtxt(filename, usecols=idx)

                cv_grid = np.reshape(cv_vector, grid_shape)
                cv_grids.append(cv_grid)

            # All fes files would generate same grids -> can break
            break

        # Compute fes grids
        fes_grids = []
        for filename in fes_files:

            fes_vector = np.loadtxt(filename,
                                    usecols=self.n_cvs)

            fes_grid = np.reshape(fes_vector, grid_shape)
            fes_grid = convert_ase_energy(fes_grid, energy_units)

            if relative:
                fes_grid -= np.min(fes_grid)

            fes_grids.append(fes_grid)

        total_cv_grid = np.stack(cv_grids, axis=0)
        total_fes_grid = np.stack(fes_grids, axis=0)

        return total_cv_grid, total_fes_grid

    def _get_min_max_params(self,
                            cvs_bounds: Optional[Sequence] = None
                            ) -> Tuple:
        """
        Compute min and max parameters for generating fes.dat files from
        HILLS.dat files.

        -----------------------------------------------------------------------
        Arguments:

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
        """

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

                extension = 0.2 * (total_max - total_min)

                min_params.append(str(total_min - extension))
                max_params.append(str(total_max + extension))

            return ','.join(min_params), ','.join(max_params)

        else:
            cvs_bounds_checked = self._check_cv_bounds(cvs_bounds)

            min_params = [str(cv_bounds[0]) for cv_bounds in cvs_bounds_checked]
            max_params = [str(cv_bounds[1]) for cv_bounds in cvs_bounds_checked]

            return ','.join(min_params), ','.join(max_params)

    def _check_cv_bounds(self,
                         cvs_bounds
                         ) -> Sequence:
        """
        Checks the validity of the supplied CVs bounds and returns the
        bounds in a universal format.

        -----------------------------------------------------------------------
        Arguments:

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,

        Returns:

            (Sequence): CVs bounds in a universal format
        """

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
