import mlptrain
import ase
import os
import re
import time
import glob
import shutil
import warnings
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import autode as ade
from typing import Optional, Sequence, Union, Tuple, List
from multiprocessing import Pool
from subprocess import Popen
from copy import deepcopy
from matplotlib.colors import ListedColormap
from scipy.stats import norm
from ase import units as ase_units
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory as ASETrajectory
from mlptrain.configurations import Configuration, ConfigurationSet
from mlptrain.sampling.md import run_mlp_md
from mlptrain.sampling.plumed import (
    PlumedBias,
    plumed_setup,
    plot_cv_versus_time,
    plot_cv1_and_cv2
)
from mlptrain.config import Config
from mlptrain.log import logger
from mlptrain.utils import (
    work_in_tmp_dir,
    unique_name,
    move_files,
    convert_ase_time,
    convert_ase_energy,
    convert_exponents
)


class Metadynamics:
    """Metadynamics class for running biased molecular dynamics using
    metadynamics bias and analysing the results"""

    def __init__(self,
                 cvs:   Union[Sequence['mlptrain._PlumedCV'],
                                       'mlptrain._PlumedCV'],
                 bias:  Optional['mlptrain.PlumedBias'] = None,
                 temp:  Optional[float] = None):
        """
        Molecular dynamics using metadynamics bias. Used for calculating free
        energies (by using well-tempered metadynamics bias) and sampling
        configurations for active learning.

        -----------------------------------------------------------------------
        Arguments:

            cvs: Sequence of PLUMED collective variables

            bias: PLUMED bias which can be supplied if additional collective
                  variables are required,
                  e.g. Metadynamics(cvs=cv1, bias=PlumedBias(cvs=(cv1, cv2)),
                  where cv1 will be biased using metadynamics, and cv2 might
                  be an additional CV with WALLS to constrain the system
        """

        if bias is not None:

            if bias.from_file:
                raise ValueError('Cannot initialise Metadynamics using '
                                 'PlumedBias initialised from a file')

            else:
                self.bias = bias

        else:
            self.bias = PlumedBias(cvs)

        self.bias._set_metad_cvs(cvs)

        self.temp = temp
        self._previous_run_parameters = {}

    @property
    def n_cvs(self) -> int:
        """Number of collective variables used in metadynamics"""
        return self.bias.n_metad_cvs

    @property
    def kbt(self) -> float:
        """Value of k_B*T in ASE units"""
        return ase_units.kB * self.temp

    def estimate_width(self,
                       configurations: Union['mlptrain.Configuration',
                                             'mlptrain.ConfigurationSet'],
                       mlp:            'mlptrain.potentials._base.MLPotential',
                       temp:           float = 300,
                       interval:       int = 10,
                       dt:             float = 1,
                       plot:           bool = True,
                       **kwargs
                       ) -> List:
        """
        Estimate optimal widths (σ) to be used in metadynamics.

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
                kwargs_single['idx'] = idx + 1

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

            pool.close()
            for width_process in width_processes:
                all_widths.append(width_process.get())
                all_widths = np.array(all_widths)
            pool.join()

        finish = time.perf_counter()
        logger.info(f'Width estimation done in {(finish - start) / 60:.1f} m')

        move_files([r'colvar_\w+_\d+\.dat'],
                   dst_folder='plumed_files/width_estimation',
                   regex=True)

        move_files([r'\w+_config\d+\.pdf'],
                   dst_folder='width_estimation',
                   regex=True)

        opt_widths = list(np.min(all_widths, axis=0))
        opt_widths_strs = []
        for cv, width in zip(self.bias.metad_cvs, opt_widths):
            if cv.units is not None:
                opt_widths_strs.append(f'{cv.name} {width:.2f} {cv.units}')

            else:
                opt_widths_strs.append(f'{cv.name} {width:.2f}')

        logger.info(f'Estimated widths: {", ".join(opt_widths_strs)}')
        return opt_widths

    def _get_width_for_single(self,
                              configuration: 'mlptrain.Configuration',
                              mlp: 'mlptrain.potentials._base.MLPotential',
                              temp: float,
                              dt: float,
                              interval: int,
                              bias: 'mlptrain.PlumedBias',
                              plot: bool,
                              **kwargs) -> List:
        """Estimate optimal widths (σ) for a single configuration"""

        logger.info(f'Running MD simulation number {kwargs["idx"]}')

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

        for cv in self.bias.metad_cvs:
            colvar_filename = f'colvar_{cv.name}_{kwargs["idx"]}.dat'

            cv_array = np.loadtxt(colvar_filename, usecols=1)

            width = np.std(cv_array)
            widths.append(width)

            if plot is True:
                plot_cv_versus_time(filename=colvar_filename,
                                    cv_units=cv.units,
                                    label=f'config{kwargs["idx"]}')

        return widths

    def run_metadynamics(self,
                         configuration: 'mlptrain.Configuration',
                         mlp:           'mlptrain.potentials._base.MLPotential',
                         temp:          float,
                         interval:      int,
                         dt:            float,
                         pace:          int = 100,
                         height:        Optional[float] = None,
                         width:         Optional = None,
                         biasfactor:    Optional[float] = None,
                         al_iter:       Optional[int] = None,
                         n_runs:        int = 1,
                         save_sep:      bool = True,
                         all_to_xyz:    bool = False,
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

            height: (float) ω, initial height of placed gaussians (in eV)

            width: (List[float] | float | None) σ, standard deviation
                   (parameter describing the width) of placed gaussians,
                   if not supplied it is estimated automatically

            biasfactor: (float | None) γ, describes how quickly gaussians
                                       shrink, larger values make gaussians to
                                       be placed less sensitive to the bias
                                       potential

            al_iter: (int | None) If not None, can be used to load metadynamics
                                  bias generated during inherited-bias active
                                  learning. The index specifies from which AL
                                  iteration to load the bias

            n_runs: (int) Number of times to run metadynamics

            save_sep: (bool) If True saves trajectories of
                             each simulation separately

            all_to_xyz: (bool) If True all .traj trajectory files are saved as
                              .xyz files (when using save_fs, save_ps, save_ns)

            restart: (bool) If True restarts the most recent metadynamics
                            simulation
        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units

            {save_fs, save_ps, save_ns}: Trajectory saving interval
                                         in some units

            {grid_min, grid_max, grid_bin}: Grid parameters: if defined, the
                                            deposited gaussians are dumped to
                                            a grid that is used to compute
                                            the biased forces

            constraints: (List) List of ASE constraints to use in the dynamics
                                e.g. [ase.constraints.Hookean(a1, a2, k, rt)]
        """

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

        if al_iter:
            self._initialise_inherited_bias(al_iter=al_iter, n_runs=n_runs)
            kwargs['load_metad_bias'] = True

        if restart:
            self._initialise_restart(width=width, n_runs=n_runs)

        else:
            if width is None:
                logger.info('Width parameters were not supplied to the '
                            'metadynamics simulation, estimating widths '
                            'automatically by performing an unbiased '
                            'simulation for 10 ps')

                width = self.estimate_width(configurations=configuration,
                                            mlp=mlp)

            kwargs['kept_substrings'] = ['.traj', '.dat']

        self.bias._set_metad_params(width=width,
                                    pace=pace,
                                    height=height,
                                    biasfactor=biasfactor,
                                    **kwargs)

        metad_processes, metad_trajs = [], []

        n_processes = min(Config.n_cores, n_runs)
        logger.info(f'Running {n_runs} independent Metadynamics '
                    f'simulation(s), {n_processes} simulation(s) run '
                    f'in parallel, 1 walker per simulation')

        start_metad = time.perf_counter()

        with Pool(processes=n_processes) as pool:

            for idx in range(n_runs):

                # Without copy kwargs is overwritten at every iteration
                kwargs_single = deepcopy(kwargs)
                kwargs_single['idx'] = idx + 1

                metad_process = pool.apply_async(func=self._run_single_metad,
                                                 args=(configuration,
                                                       mlp,
                                                       temp,
                                                       interval,
                                                       dt,
                                                       self.bias,
                                                       al_iter,
                                                       restart),
                                                 kwds=kwargs_single)
                metad_processes.append(metad_process)

            pool.close()
            for metad_process in metad_processes:
                metad_trajs.append(metad_process.get())
            pool.join()

        finish_metad = time.perf_counter()
        logger.info('Metadynamics done in '
                    f'{(finish_metad - start_metad) / 60:.1f} m')

        # Move .traj files into 'trajectories' folder and compute .xyz files
        self._move_and_save_files(metad_trajs=metad_trajs,
                                  save_sep=save_sep,
                                  all_to_xyz=all_to_xyz,
                                  restart=restart)
        if al_iter is None:
            self.plot_gaussian_heights()

        self._set_previous_parameters(configuration=configuration,
                                      mlp=mlp,
                                      temp=temp,
                                      dt=dt,
                                      interval=interval,
                                      **kwargs)
        return None

    @staticmethod
    def _initialise_inherited_bias(al_iter: int, n_runs: int) -> None:
        """
        Initialise files in order to load a bias generated during
        inherited-bias active learning
        """

        bias_path = f'accumulated_bias/bias_after_iter_{al_iter}.dat'
        if os.path.exists(bias_path):
            for idx in range(n_runs):
                shutil.copyfile(src=bias_path, dst=f'HILLS_{idx+1}.dat')

        else:
            raise FileNotFoundError('Inherited bias generated after AL '
                                    f'iteration {al_iter} not found')

        return None

    def _initialise_restart(self, width: Sequence, n_runs: int) -> None:
        """
        Initialise restart for metadynamics simulation by checking
        conditions and moving files
        """

        if width is None:
            raise ValueError('Make sure to use exactly the same width as '
                             'in the previous simulation')

        if not os.path.exists('plumed_files/metadynamics'):
            raise FileNotFoundError('Metadynamics folder not found, make '
                                    'sure to run metadynamics before '
                                    'trying to restart')

        metad_path = os.path.join(os.getcwd(), 'plumed_files/metadynamics')
        traj_path = os.path.join(os.getcwd(), 'trajectories')

        for cv in self.bias.metad_cvs:
            colvar_path = os.path.join(metad_path, f'colvar_{cv.name}_*.dat')
            n_previous_runs = len(glob.glob(colvar_path))
            if n_previous_runs != n_runs:
                raise NotImplementedError('Restart is implemented only if the '
                                          'number of runs matches the number '
                                          'of runs in the previous simulation')

        for filename in glob.glob(os.path.join(metad_path, 'fes_*.dat')):
            os.remove(filename)

        move_files(['.dat'],
                   dst_folder=os.getcwd(),
                   src_folder=metad_path,
                   unique=False)

        move_files(['.traj'],
                   dst_folder=os.getcwd(),
                   src_folder=traj_path,
                   unique=False)

        return None

    def _run_single_metad(self,
                          configuration: 'mlptrain.Configuration',
                          mlp: 'mlptrain.potentials._base.MLPotential',
                          temp: float,
                          interval: int,
                          dt: float,
                          bias: 'mlptrain.PlumedBias',
                          al_iter: Optional[int] = None,
                          restart: Optional[bool] = False,
                          **kwargs) -> 'mlptrain.Trajectory':
        """Initiate a single metadynamics run"""

        logger.info('Running Metadynamics simulation '
                    f'number {kwargs["idx"]}')

        if al_iter is not None:
            kwargs['copied_substrings'] = [f'HILLS_{kwargs["idx"]}.dat']

        if restart:
            restart_files = []

            for cv in self.bias.cvs:
                restart_files.append(f'colvar_{cv.name}_{kwargs["idx"]}.dat')

            restart_files.append(f'HILLS_{kwargs["idx"]}.dat')
            restart_files.append(f'trajectory_{kwargs["idx"]}.traj')

        else:
            restart_files = None

        kwargs['n_cores'] = 1

        traj = run_mlp_md(configuration=configuration,
                          mlp=mlp,
                          temp=temp,
                          dt=dt,
                          interval=interval,
                          bias=bias,
                          restart_files=restart_files,
                          **kwargs)

        return traj

    @staticmethod
    def _move_and_save_files(metad_trajs: List['mlptrain.Trajectory'],
                             save_sep: bool,
                             all_to_xyz: bool,
                             restart: bool
                             ) -> None:
        """
        Save metadynamics trajectories, move them into trajectories folder
        and compute .xyz files
        """

        move_files(['.dat'],
                   dst_folder='plumed_files/metadynamics',
                   unique=True if not restart else False)

        move_files([r'trajectory_\d+\.traj', r'trajectory_\d+_\w+\.traj'],
                   dst_folder='trajectories',
                   regex=True,
                   unique=True if not restart else False)

        os.chdir('trajectories')

        if save_sep:
            for idx, metad_traj in enumerate(metad_trajs, start=1):
                metad_traj.save(filename=f'metad_{idx}.xyz')

        else:
            combined_traj = ConfigurationSet(allow_duplicates=True)
            for metad_traj in metad_trajs:
                combined_traj += metad_traj

            combined_traj.save(filename='combined_trajectory.xyz')

        if all_to_xyz:
            pattern = re.compile(r'trajectory_\d+_\w+\.traj')

            for filename in os.listdir():
                if re.search(pattern, filename) is not None:
                    basename = filename[:-5]
                    idx = basename.split('_')[1]
                    sim_time = basename.split('_')[2]

                    ase_traj = ASETrajectory(filename)
                    ase_write(f'metad_{idx}_{sim_time}.xyz', ase_traj)

        os.chdir('..')

        return None

    def _set_previous_parameters(self,
                                 configuration: 'mlptrain.Configuration',
                                 mlp: 'mlptrain.potentials._base.MLPotential',
                                 temp: float,
                                 dt: float,
                                 interval: int,
                                 **kwargs) -> None:
        """Set parameters in the _previous_run_parameters"""

        self._previous_run_parameters = {'configuration': configuration,
                                         'mlp': mlp,
                                         'temp': temp,
                                         'dt': dt,
                                         'interval': interval}

        sim_time_dict = {}
        for key in ['ps', 'fs', 'ns']:
            if key in kwargs:
                sim_time_dict[key] = kwargs[key]

        self._previous_run_parameters['sim_time_dict'] = sim_time_dict

        return None

    def plot_gaussian_heights(self,
                              energy_units: str = 'kcal mol-1',
                              time_units:   str = 'ps',
                              path:         str = 'plumed_files/metadynamics'
                              ) -> None:
        """
        Plot the height of deposited gaussians as a function of time (using
        HILLS_{idx}.dat files).

        -----------------------------------------------------------------------
        Arguments:

            energy_units: (str) Energy units to be used in plotting, available
                                units: 'eV', 'kcal mol-1', 'kJ mol-1'

            time_units: (str) Time units to be used in plotting, available
                              units: 'fs', 'ps', 'ns'

            path: (str) Directory where HILLS_{idx}.dat files are located
        """

        if not os.path.exists(path):
            raise FileNotFoundError('Directory with metadynamics files not '
                                    'found. Make sure to run metadynamics '
                                    'before using this method')

        initial_path = os.getcwd()
        os.chdir(path)

        idx = 1
        while os.path.exists(f'HILLS_{idx}.dat'):
            self._plot_gaussian_heights_single(idx=idx,
                                               energy_units=energy_units,
                                               time_units=time_units)
            idx += 1

        os.chdir(initial_path)
        move_files([r'gaussian_heights_\d+.pdf'],
                   src_folder=path,
                   dst_folder='gaussian_heights',
                   regex=True)

        return None

    @staticmethod
    def _plot_gaussian_heights_single(idx:          int,
                                      energy_units: str = 'kcal mol-1',
                                      time_units:   str = 'ps'
                                      ) -> None:
        """
        Plot the height of deposited gaussians as a function of time for a
        single metadynamics run.

        -----------------------------------------------------------------------
        Arguments:

            idx: (int) Index specifying metadynamics run number

            energy_units: (str) Energy units to be used in plotting, available
                                units: 'eV', 'kcal mol-1', 'kJ mol-1'

            time_units: (str) Time units to be used in plotting, available
                              units: 'fs', 'ps', 'ns'
        """

        filename = f'HILLS_{idx}.dat'

        deposit_time = np.loadtxt(filename, usecols=0)
        deposit_time = convert_ase_time(time_array=deposit_time,
                                        units=time_units)

        heights = np.loadtxt(filename, usecols=-2)
        heights = convert_ase_energy(energy_array=heights, units=energy_units)

        fig, ax = plt.subplots()
        ax.plot(deposit_time, heights)

        ax.set_xlabel(f'Time / {time_units}')
        ax.set_ylabel(f'Gaussian height / {convert_exponents(energy_units)}')

        fig.tight_layout()
        fig.savefig(f'gaussian_heights_{idx}.pdf')
        plt.close(fig)

        return None

    def try_multiple_biasfactors(self,
                                 configuration: 'mlptrain.Configuration',
                                 mlp:  'mlptrain.potentials._base.MLPotential',
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
        Execute multiple well-tempered metadynamics runs in parallel with a
        provided sequence of biasfactors and plot the resulting trajectories,
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

            height: (float) ω, initial height of placed gaussians (in eV)

            width: (List[float] | float | None) σ, standard deviation
                   (parameter describing the width) of placed gaussians,
                   if not supplied it is estimated automatically

            plotted_cvs: Sequence of one or two PlumedCV objects which are
                         going to be plotted, must be a subset for CVs used to
                         define the Metadynamics object

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units

            {grid_min, grid_max, grid_bin}: Grid parameters: if defined, the
                                            deposited gaussians are dumped to
                                            a grid that is used to compute
                                            the biased forces

            constraints: (List) List of ASE constraints to use in the dynamics
                                e.g. [ase.constraints.Hookean(a1, a2, k, rt)]
        """

        logger.info(f'Trying {len(biasfactors)} different biasfactors')

        if not isinstance(biasfactors, Sequence):
            raise TypeError('Supplied biasfactors variable must be a sequence')

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
                        'automatically by performing an unbiased simulation'
                        'for 10 ps')

            width = self.estimate_width(configurations=configuration,
                                        mlp=mlp)

        # Dummy bias which stores CVs, useful for checking CVs input
        if plotted_cvs is not None:
            cvs_holder = PlumedBias(plotted_cvs)
            cvs_holder._set_metad_cvs(plotted_cvs)
        else:
            cvs_holder = self.bias

        if cvs_holder.n_cvs > 2:
            raise NotImplementedError('Plotting using more than two CVs is '
                                      'not implemented')

        if not all(cv in self.bias.metad_cvs for cv in cvs_holder.metad_cvs):
            raise ValueError('At least one of the supplied CVs are not within '
                             'the set of CVs used to define the Metadynamics '
                             'object')

        self.bias._set_metad_params(width=width,
                                    pace=pace,
                                    height=height,
                                    **kwargs)

        n_processes = min(Config.n_cores, len(biasfactors))
        logger.info('Running Well-Tempered Metadynamics simulations '
                    f'with {len(biasfactors)} different biasfactors, '
                    f'{n_processes} simulation(s) run in parallel, '
                    f'1 walker per simulation')

        start = time.perf_counter()

        with mp.get_context('spawn').Pool(processes=n_processes) as pool:

            for idx, biasfactor in enumerate(biasfactors):

                bias = deepcopy(self.bias)
                bias.biasfactor = biasfactor

                kwargs_single = deepcopy(kwargs)
                kwargs_single['idx'] = idx + 1

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

        finish = time.perf_counter()
        logger.info('Simulations with multiple biasfactors done in '
                    f'{(finish - start) / 60:.1f} m')

        move_files([r'colvar_\w+_\d+\.dat', r'HILLS_\d+\.dat'],
                   dst_folder='plumed_files/multiple_biasfactors',
                   regex=True)

        move_files([r'\w+_biasf\d+\.pdf'],
                   dst_folder='multiple_biasfactors',
                   regex=True)

        return None

    def _try_single_biasfactor(self,
                               configuration: 'mlptrain.Configuration',
                               mlp: 'mlptrain.potentials._base.MLPotential',
                               temp: float,
                               interval: int,
                               dt: float,
                               bias: 'mlptrain.PlumedBias',
                               plotted_cvs: Optional,
                               **kwargs):
        """
        Execute a single well-tempered metadynamics run and plot the
        resulting trajectory
        """

        self._run_single_metad(configuration=configuration,
                               mlp=mlp,
                               temp=temp,
                               interval=interval,
                               dt=dt,
                               bias=bias,
                               kept_substrings=['.dat'],
                               **kwargs)

        filenames = [f'colvar_{cv.name}_{kwargs["idx"]}.dat'
                     for cv in plotted_cvs]

        for filename, cv in zip(filenames, plotted_cvs):
            plot_cv_versus_time(filename=filename,
                                cv_units=cv.units,
                                label=f'biasf{bias.biasfactor}')

        if len(plotted_cvs) == 2:
            plot_cv1_and_cv2(filenames=filenames,
                             cvs_units=[cv.units for cv in plotted_cvs],
                             label=f'biasf{bias.biasfactor}')

        return None

    def block_analysis(self,
                       start_time:         float,
                       idx:                int = 1,
                       energy_units:       str = 'kcal mol-1',
                       n_bins:             int = 300,
                       min_n_blocks:       int = 10,
                       min_blocksize:      int = 10,
                       blocksize_interval: int = 10,
                       bandwidth:          float = 0.02,
                       cvs_bounds:         Optional[Sequence] = None,
                       temp:               Optional[float] = None,
                       dt:                 Optional[float] = None,
                       interval:           Optional[int] = None,
                       ) -> None:
        """
        Perform block averaging analysis on the sliced trajectory of the most
        recent metadynamics run. Plot the block analysis and save mean FES
        grids with a range of block sizes, which, if the block analysis
        converged, can be used for plotting the FES error using plot_fes()
        method.

        -----------------------------------------------------------------------
        Arguments:

            start_time: (float) Start time of the sliced trajectory which is
                                going to be used for block averaging analysis
                                (in ps)

            idx: (int) Index specifying which metadynamics run (from n_runs)
                       to use for block analysis

            energy_units: (str) Energy units to be used in plotting, available
                                units: 'eV', 'kcal mol-1', 'kJ mol-1'

            n_bins: (int) Number of bins to use when dumping histograms

            min_n_blocks: (int) Minimum number of blocks

            min_blocksize: (int) Minimum size of a block

            blocksize_interval: (int) Block size interval describing the
                                      difference in the number of frames per
                                      block for adjacent block sizes

            bandwidth: (float) Bandwidth parameter used in the histogram
                               estimations

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
                        e.g. [(0.8, 1.5), (80, 120)]

            temp: (float) Temperature in K during the analysed simulation

            dt: (float) Time-step in fs during the analysed simulation

            interval: (int) Interval between saving the geometry during the
                            analysed simulation
        """

        start = time.perf_counter()

        bias, temp, dt, interval = self._reweighting_params(temp=temp,
                                                            dt=dt,
                                                            interval=interval)
        start_frame_index = int((start_time * 1E3) / (dt * interval))

        sliced_traj = ase_read(f'trajectories/trajectory_{idx}.traj',
                               index=f'{start_frame_index}:')
        self._save_ase_traj_as_xyz(sliced_traj)

        shutil.copyfile(src=f'plumed_files/metadynamics/HILLS_{idx}.dat',
                        dst=f'HILLS_{idx}.dat')

        # Writes plumed_setup.dat
        plumed_setup(bias=bias,
                     temp=temp,
                     interval=interval,
                     idx=idx,
                     load_metad_bias=True,
                     remove_print=True,
                     write_plumed_setup=True)

        min_max_params = self._get_min_max_params(cvs_bounds=cvs_bounds,
                                                  path='plumed_files/'
                                                       'metadynamics')

        # The number of frames PLUMED driver takes into account
        # n_used_frames = n_total_frames - 1
        n_used_frames = len(sliced_traj) - 1
        max_blocksize = n_used_frames // min_n_blocks

        if max_blocksize < min_blocksize:
            raise ValueError('The simulation is too short to perform '
                             'block analysis')

        logger.info('Performing block analysis in parallel using '
                    f'{Config.n_cores} cores')

        grid_procs, data_dict = [], {}
        blocksizes = list(range(min_blocksize, max_blocksize + 1,
                                blocksize_interval))

        with Pool(processes=Config.n_cores) as pool:

            for blocksize in blocksizes:

                grid_proc = pool.apply_async(func=self._compute_grids_for_blocksize,
                                             args=(blocksize,
                                                   temp,
                                                   min_max_params,
                                                   n_bins,
                                                   bandwidth,
                                                   energy_units))
                grid_procs.append(grid_proc)

            pool.close()
            for blocksize, grid_proc in zip(blocksizes, grid_procs):
                cvs_grid, fes_error = grid_proc.get()
                data_dict[str(blocksize)] = fes_error
            pool.join()

        data_dict['CVs'] = cvs_grid
        np.savez('block_analysis.npz', **data_dict)

        os.remove('plumed_setup.dat')
        os.remove('traj.xyz')
        os.remove(f'HILLS_{idx}.dat')

        self._plot_block_analysis(blocksizes=blocksizes,
                                  data_dict=data_dict,
                                  energy_units=energy_units)

        finish = time.perf_counter()
        logger.info(f'Block analysis done in {(finish - start) / 60:.1f} m')

        return None

    def _reweighting_params(self,
                            temp: float,
                            dt: float,
                            interval: int
                            ) -> Tuple:
        """
        Read parameters required for reweighting from the previous
        metadynamics simulation. If previous parameters are not set, read
        parameters which are supplied to the method
        """

        # Bias with dummy width and height values, and very large pace
        bias = deepcopy(self.bias)
        bias._set_metad_params(pace=int(1E9),
                               width=[1 for _ in range(self.n_cvs)],
                               height=0)

        _parameters = [temp, dt, interval]

        if len(self._previous_run_parameters) != 0:
            temp = self._previous_run_parameters['temp']
            dt = self._previous_run_parameters['dt']
            interval = self._previous_run_parameters['interval']

        elif any(param is None for param in _parameters):

            raise TypeError('Metadynamics object does not have all the '
                            'required parameters to run block analysis. '
                            'Please provide parameters from the previous '
                            'metadynamics run')

        return bias, temp, dt, interval

    @staticmethod
    def _save_ase_traj_as_xyz(ase_traj: 'ase.io.trajectory.Trajectory'):
        """Save ASE trajectory as .xyz file"""

        _mlt_configuration_set = ConfigurationSet(allow_duplicates=True)
        for atoms in ase_traj:
            config = Configuration()
            config.atoms = [ade.Atom(label) for label in atoms.symbols]

            for i, position in enumerate(atoms.get_positions()):
                config.atoms[i].coord = position

            _mlt_configuration_set.append(config)

        _mlt_configuration_set.save('traj.xyz')

        return None

    @work_in_tmp_dir(copied_substrings=['traj.xyz',
                                        'plumed_setup.dat',
                                        'HILLS'])
    def _compute_grids_for_blocksize(self,
                                     blocksize: int,
                                     temp: float,
                                     min_max_params: Tuple,
                                     n_bins: int,
                                     bandwidth: float,
                                     energy_units: str
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute CV and FES error grids over blocks for a given block size and
        return both grids"""

        self._generate_hist_files_by_reweighting(blocksize=blocksize,
                                                 temp=temp,
                                                 min_max_params=min_max_params,
                                                 n_bins=n_bins,
                                                 bandwidth=bandwidth)

        normal, hist, cvs_grid = self._read_histogram(filename='hist.dat',
                                                      n_bins=n_bins,
                                                      compute_cvs=True)
        normal_sq = normal**2
        average = normal * hist
        average_sq = normal * hist**2

        for hist_file in glob.glob('analysis.*.hist.dat'):
            tnormal, new_hist, _ = self._read_histogram(filename=hist_file,
                                                        n_bins=n_bins)
            normal += tnormal
            normal_sq += tnormal**2
            average += tnormal * new_hist
            average_sq += tnormal * new_hist**2

        average /= normal
        variance = (average_sq / normal) - average**2
        variance *= (normal / (normal - (normal_sq / normal)))

        n_grids = 1 + len(glob.glob('analysis.*.hist.dat'))
        hist_error = np.sqrt(variance / n_grids)

        fes_error = ase_units.kB * temp * hist_error
        fes_error = np.divide(fes_error, average,
                              out=np.zeros_like(fes_error),
                              where=average != 0)
        fes_error = convert_ase_energy(energy_array=fes_error,
                                       units=energy_units)

        return cvs_grid, fes_error

    def _read_histogram(self,
                        filename:    str,
                        n_bins:      int,
                        compute_cvs: bool = False
                        ) -> Tuple:
        """
        Read the histogram file and return the normalisation together with
        the CVs and the histogram as numpy grids
        """

        grid_shape = tuple([n_bins for _ in range(self.n_cvs)])

        hist_vector = np.loadtxt(filename, usecols=self.n_cvs)
        hist = np.reshape(hist_vector, (1, *grid_shape))

        cvs_grid = np.zeros((self.n_cvs, *grid_shape))
        if compute_cvs:
            for idx in range(self.n_cvs):
                cv_vector = np.loadtxt(filename, usecols=idx)

                cv_grid = np.reshape(cv_vector, grid_shape)
                cvs_grid[idx] = cv_grid

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("#! SET normalisation"):
                    normal = line.split()[3]
                    break

        return float(normal), hist, cvs_grid

    def _generate_hist_files_by_reweighting(self,
                                            blocksize: Optional[int],
                                            temp: float,
                                            min_max_params: Tuple,
                                            n_bins: int,
                                            bandwidth: float
                                            ) -> None:
        """
        Generate analysis.*.hist.dat + hist.dat files which are required
        for block analysis
        """

        os.environ['PLUMED_MAXBACKUP'] = '10000'

        min_param_seq, max_param_seq = min_max_params
        bandwidth_seq = ','.join(str(bandwidth) for _ in range(self.n_cvs))
        bin_param_seq = ','.join(str(n_bins-1) for _ in range(self.n_cvs))

        clear_setup = f'CLEAR={blocksize} ' if blocksize is not None else ''
        stride_setup = f'STRIDE={blocksize} ' if blocksize is not None else ''

        reweight_setup = ['as: REWEIGHT_BIAS '
                          f'TEMP={temp} '
                          'ARG=metad.bias',
                          'hist: HISTOGRAM '
                          f'ARG={self.bias.metad_cv_sequence} '
                          f'STRIDE=1 '
                          f'{clear_setup}'
                          f'GRID_MIN={min_param_seq} '
                          f'GRID_MAX={max_param_seq} '
                          f'GRID_BIN={bin_param_seq} '
                          f'BANDWIDTH={bandwidth_seq} '
                          'LOGWEIGHTS=as',
                          'DUMPGRID '
                          'GRID=hist '
                          f'{stride_setup}'
                          'FILE=hist.dat']

        os.rename('plumed_setup.dat', 'reweight.dat')
        with open('reweight.dat', 'a') as f:
            for line in reweight_setup:
                f.write(f'{line}\n')

        self.bias.write_cv_files()

        driver_process = Popen(['plumed', 'driver',
                                '--ixyz', 'traj.xyz',
                                '--plumed', 'reweight.dat',
                                '--length-units', 'A'])
        driver_process.wait()

        return None

    @staticmethod
    def _plot_block_analysis(blocksizes: List,
                             data_dict: dict,
                             energy_units: str
                             ) -> None:
        """Plot the standard deviation versus block size"""

        data_dict.pop('CVs')
        mean_errors = [np.mean(error_grid) for error_grid in data_dict.values()]

        fig, ax = plt.subplots()
        ax.plot(blocksizes, mean_errors, color='k')

        ax.set_xlabel('Block size')
        ax.set_ylabel(r'$\left\langle\sigma_{G}\right\rangle$ / '
                      f'{convert_exponents(energy_units)}')

        fig.tight_layout()

        figname = 'block_analysis.pdf'
        if os.path.exists(figname):
            os.rename(figname, unique_name(figname))

        fig.savefig(figname)
        plt.close(fig)

        return None

    def plot_fes(self,
                 energy_units:      str = 'kcal mol-1',
                 confidence_level:  float = 0.95,
                 n_bins:            int = 300,
                 cvs_bounds:        Optional[Sequence] = None,
                 fes_npy:           Optional[str] = None,
                 blocksize:         Optional[int] = None,
                 ) -> None:
        """
        Plot the free energy surface with a confidence interval. If the .npy
        file is not supplied, the file is computed (if metadynamics has been
        run). If blocksize is supplied, the FES error is plotted using
        block_analysis.npz file with the given blocksize generated during
        block averaging analysis.

        -----------------------------------------------------------------------
        Arguments:

            energy_units: (str) Energy units to be used in computing the free
                                energy, available units: 'eV', 'kcal mol-1',
                                'kJ mol-1'. If fes_npy and/or blocksize options
                                are used, the units of fes_npy and/or blocksize
                                files must match energy units here

            confidence_level: (float) Specifies what confidence level to use
                                      in plots (probability for FES to lie in
                                      the plotted range)

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
                        e.g. [(0.8, 1.5), (80, 120)]

            fes_npy: (str) File name of the .npy file used for plotting. FES
                           obtained through block analysis should be specified
                           using blocksize

            blocksize: (int) If block analysis has been performed, the integer
                             specifies which block size to use for plotting
                             the FES standard error
        """

        if fes_npy is None:
            if not os.path.exists('fes_raw.npy'):
                fes = self.compute_fes(energy_units, n_bins, cvs_bounds)

            else:
                logger.info('Using fes_raw.npy in the current directory for '
                            'plotting')

                fes = np.load('fes_raw.npy')

        else:
            fes = np.load(fes_npy)

        if self.n_cvs == 1:
            self._plot_1d_fes(fes=fes,
                              energy_units=energy_units,
                              confidence_level=confidence_level,
                              blocksize=blocksize)

        elif self.n_cvs == 2:
            self._plot_2d_fes(fes=fes,
                              energy_units=energy_units,
                              confidence_level=confidence_level,
                              blocksize=blocksize)

        else:
            raise NotImplementedError('Plotting FES is available only for one '
                                      'and two collective variables')

        return None

    def compute_fes(self,
                    energy_units:    str = 'kcal mol-1',
                    n_bins:          int = 300,
                    cvs_bounds:      Optional[Sequence] = None,
                    via_reweighting: bool = False,
                    start_time:      float = 0.00,
                    bandwidth:       float = 0.02,
                    temp:            Optional[float] = None,
                    dt:              Optional[float] = None,
                    interval:        Optional[int] = None,
                    ) -> np.ndarray:
        """
        Compute a grid containing collective variable grids and free energy
        surface grids, which is saved in the current directory as .npy file.

        -----------------------------------------------------------------------
        Arguments:

            energy_units: (str) Energy units to be used in plotting, available
                                units: 'eV', 'kcal mol-1', 'kJ mol-1'

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable
                        e.g. [(0.8, 1.5), (80, 120)]

            via_reweighting: (bool) If true the free energy is computed using
                                    the reweighting scheme

            start_time: (float) Start time of the sliced trajectory which is
                                going to be used in reweighting (in ps)

            bandwidth: (float) Bandwidth parameter used in the histogram
                               estimations

            temp: (float) Temperature in K during the analysed simulation,
                          only needed when reweighting

            dt: (float) Time-step in fs during the analysed simulation, only
                        needed when reweighting

            interval: (int) Interval between saving the geometry during the
                            analysed simulation, only needed
                            when reweighting

        Returns:

            (np.ndarray): The total grid containing CVs and FESs
        """

        logger.info('Computing and saving the free energy grid as fes_raw.npy')

        if via_reweighting:
            fes_raw = self._compute_fes_via_reweighting(energy_units=energy_units,
                                                        n_bins=n_bins,
                                                        cvs_bounds=cvs_bounds,
                                                        temp=temp,
                                                        dt=dt,
                                                        interval=interval,
                                                        start_time=start_time,
                                                        bandwidth=bandwidth)
        else:
            fes_raw = self._compute_fes_via_hills(energy_units=energy_units,
                                                  n_bins=n_bins,
                                                  cvs_bounds=cvs_bounds)
        np.save('fes_raw.npy', fes_raw)
        return fes_raw

    def _compute_fes_via_reweighting(self,
                                     temp: float,
                                     dt: float,
                                     interval: int,
                                     start_time: float,
                                     energy_units: str,
                                     cvs_bounds: Optional[Sequence],
                                     n_bins: int,
                                     bandwidth: float
                                     ) -> np.ndarray:
        """
        Compute the free energy surface grid by reweighting the biased
        distribution
        """

        bias, temp, dt, interval = self._reweighting_params(temp=temp,
                                                            dt=dt,
                                                            interval=interval)
        start_frame_index = int((start_time * 1E3) / (dt * interval))

        min_max_params = self._get_min_max_params(cvs_bounds=cvs_bounds,
                                                  path='plumed_files/'
                                                       'metadynamics')
        n_runs = len(glob.glob('trajectories/trajectory_*.traj'))
        n_processes = min(Config.n_cores, n_runs)
        fes_processes, fes_grids = [], []

        with Pool(processes=n_processes) as pool:
            for idx in range(1, n_runs+1):
                traj_path = os.path.join(os.getcwd(),
                                         f'trajectories/trajectory_{idx}.traj')
                hills_path = os.path.join(os.getcwd(),
                                          f'plumed_files/metadynamics/HILLS_{idx}.dat')

                proc = pool.apply_async(func=self._compute_single_fes_via_reweighting,
                                        args=(idx,
                                              traj_path,
                                              hills_path,
                                              start_frame_index,
                                              bias,
                                              temp,
                                              interval,
                                              min_max_params,
                                              n_bins,
                                              bandwidth,
                                              energy_units))
                fes_processes.append(proc)

            pool.close()
            for proc in fes_processes:
                cvs_grid, fes_grid = proc.get()
                fes_grids.append(fes_grid)
            pool.join()

        fes_raw = np.concatenate([cvs_grid] + fes_grids, axis=0)
        return fes_raw

    @work_in_tmp_dir()
    def _compute_single_fes_via_reweighting(self,
                                            idx: int,
                                            traj_path: str,
                                            hills_path: str,
                                            start_frame_index: int,
                                            bias: 'mlptrain.PlumedBias',
                                            temp: float,
                                            interval: int,
                                            min_max_params: Tuple,
                                            n_bins: int,
                                            bandwidth: float,
                                            energy_units: str
                                            ) -> Tuple:
        """Compute CVs and FES grids for a single run by reweighting"""

        sliced_traj = ase_read(traj_path, index=f'{start_frame_index}:')
        self._save_ase_traj_as_xyz(sliced_traj)

        shutil.copyfile(src=hills_path, dst=f'HILLS_{idx}.dat')
        plumed_setup(bias=bias,
                     temp=temp,
                     interval=interval,
                     idx=idx,
                     load_metad_bias=True,
                     remove_print=True,
                     write_plumed_setup=True)

        self._generate_hist_files_by_reweighting(blocksize=None,
                                                 temp=temp,
                                                 min_max_params=min_max_params,
                                                 n_bins=n_bins,
                                                 bandwidth=bandwidth)

        _, hist, cvs_grid = self._read_histogram(filename='hist.dat',
                                                 n_bins=n_bins,
                                                 compute_cvs=True)

        fes_grid = -ase_units.kB * temp * np.log(hist)
        fes_grid = fes_grid - np.min(fes_grid)
        fes_grid = convert_ase_energy(energy_array=fes_grid,
                                      units=energy_units)

        return cvs_grid, fes_grid

    def _compute_fes_via_hills(self,
                               energy_units: str = 'kcal mol-1',
                               n_bins:       int = 300,
                               cvs_bounds:   Optional[Sequence] = None,
                               ) -> np.ndarray:
        """Compute the free energy surface grid using the deposited bias"""

        try:
            os.chdir('plumed_files/metadynamics')
        except FileNotFoundError:
            raise FileNotFoundError('Metadynamics directory not found. Make '
                                    'sure to run metadynamics before trying '
                                    'to compute the FES')

        fes_files = [fname for fname in os.listdir() if fname.startswith('fes')]
        for fname in fes_files:
            os.remove(fname)

        self._generate_fes_files(n_bins=n_bins, cvs_bounds=cvs_bounds)

        cv_grids, fes_grids = self._fes_files_to_grids(energy_units=energy_units,
                                                       n_bins=n_bins)

        os.chdir('../..')

        fes_raw = np.concatenate((cv_grids, fes_grids), axis=0)
        return fes_raw

    def _plot_1d_fes(self,
                     fes:               np.ndarray,
                     energy_units:      str = 'kcal mol-1',
                     confidence_level:  float = 0.95,
                     blocksize:         Optional[int] = None,
                     ) -> None:
        """Plot 1D mean free energy surface with a confidence interval"""

        logger.info('Plotting 1D FES')

        cv_grid = fes[0]
        fes_grids = fes[1:]

        fig, ax = plt.subplots()

        mean_fes = np.mean(fes_grids, axis=0)
        ax.plot(cv_grid, mean_fes, label='Free energy')

        fes_error = self._compute_fes_error(fes_grids=fes_grids,
                                            blocksize=blocksize)
        if np.any(fes_error):

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',
                                        message='invalid value encountered in multiply')
                confidence_interval = norm.interval(confidence_level,
                                                    loc=mean_fes,
                                                    scale=fes_error)

            lower_bound = confidence_interval[0]
            upper_bound = confidence_interval[1]

            ax.fill_between(cv_grid, lower_bound, upper_bound,
                            alpha=0.3,
                            label='Confidence interval')

        cv = self.bias.metad_cvs[0]
        if cv.units is not None:
            ax.set_xlabel(f'{cv.name} / {cv.units}')

        else:
            ax.set_xlabel(f'{cv.name}')

        ax.set_ylabel(r'$\Delta G$ / '
                      f'{convert_exponents(energy_units)}')

        ax.legend()
        fig.tight_layout()

        figname = 'metad_free_energy.pdf'
        if os.path.exists(figname):
            os.rename(figname, unique_name(figname))

        fig.savefig(figname)
        plt.close(fig)

        return None

    def _plot_2d_fes(self,
                     fes:               np.ndarray,
                     energy_units:      str = 'kcal mol-1',
                     confidence_level:  float = 0.95,
                     blocksize:         Optional[int] = None,
                     ) -> None:
        """Plot 2D mean free energy surface with a confidence interval"""

        logger.info('Plotting 2D FES')

        cv1_grid = fes[0]
        cv2_grid = fes[1]
        fes_grids = fes[2:]

        fig, (ax_mean, ax_std_error) = plt.subplots(nrows=1,
                                                    ncols=2,
                                                    figsize=(12, 5))

        jet_cmap = plt.get_cmap('jet')
        jet_cmap_matrix = jet_cmap(np.linspace(0, 1, 256))
        jet_cmap_matrix[-1, :] = [1, 1, 1, 1]
        mod_jet_cmap = ListedColormap(jet_cmap_matrix)

        mean_fes = np.mean(fes_grids, axis=0)

        mean_contourf = ax_mean.contourf(cv1_grid, cv2_grid, mean_fes, 256,
                                         cmap=mod_jet_cmap)
        ax_mean.contour(cv1_grid, cv2_grid, mean_fes, 20,
                        colors='k',
                        alpha=0.2)

        mean_cbar = fig.colorbar(mean_contourf, ax=ax_mean)
        mean_cbar.set_label(label=r'$\Delta G$ / '
                                  f'{convert_exponents(energy_units)}')

        fes_error = self._compute_fes_error(fes_grids=fes_grids,
                                            blocksize=blocksize)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='invalid value encountered in multiply')
            confidence_interval = norm.interval(confidence_level,
                                                loc=mean_fes,
                                                scale=fes_error)

        interval_range = confidence_interval[1] - confidence_interval[0]

        std_error_contourf = ax_std_error.contourf(cv1_grid, cv2_grid,
                                                   interval_range, 256,
                                                   cmap='Blues')
        ax_std_error.contour(cv1_grid, cv2_grid, mean_fes, 20,
                             colors='k',
                             alpha=0.2)

        std_error_cbar = fig.colorbar(std_error_contourf, ax=ax_std_error)
        std_error_cbar.set_label(label='Confidence interval / '
                                       f'{convert_exponents(energy_units)}')

        cv1 = self.bias.metad_cvs[0]
        cv2 = self.bias.metad_cvs[1]
        for ax in (ax_mean, ax_std_error):

            if cv1.units is not None:
                ax.set_xlabel(f'{cv1.name} / {cv1.units}')

            else:
                ax.set_xlabel(f'{cv1.name}')

            if cv2.units is not None:
                ax.set_ylabel(f'{cv2.name} / {cv2.units}')

            else:
                ax.set_ylabel(f'{cv2.name}')

        for c in mean_contourf.collections:
            c.set_edgecolor("face")
        for c in std_error_contourf.collections:
            c.set_edgecolor("face")

        fig.tight_layout()

        figname = 'metad_free_energy.pdf'
        if os.path.exists(figname):
            os.rename(figname, unique_name(figname))

        fig.savefig(figname)
        plt.close(fig)

        return None

    @staticmethod
    def _compute_fes_error(fes_grids: np.ndarray,
                           blocksize: Optional[int] = None
                           ) -> np.ndarray:
        """Compute standard error of the free energy to use in plotting"""

        n_surfaces = len(fes_grids)

        if blocksize is not None:
            try:
                fes_error = np.load('block_analysis.npz')[str(blocksize)]
                fes_error = fes_error.flatten()

            except (FileNotFoundError, KeyError):
                raise FileNotFoundError('Block averaging analysis with block '
                                        f'size {blocksize} was not found. '
                                        'Make sure to run block analysis '
                                        'before using this option and use an '
                                        'appropriate block size')

        elif n_surfaces != 1:
            fes_error = ((1 / np.sqrt(n_surfaces))
                         * np.std(fes_grids, axis=0, ddof=1))

        else:
            fes_error = np.zeros_like(fes_grids[0])

        return fes_error

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
        Compute multiple fes.dat files from a HILLS_idx.dat file by summing
        the deposited gaussians using a stride. Use the computed files to plot
        multiple FESs as a function of simulation time.

        -----------------------------------------------------------------------
        Arguments:

            stride: (int) Interval that specifies the number of new gaussians
                          that is put into subsequent fes_idx_*.dat file

            n_surfaces: (int) Number of surfaces to be plotted (counting from
                              the last computed surface), must not exceed the
                              number of computed surfaces

            time_units: (str) Time units to be used in plotting, available
                              units: 'fs', 'ps', 'ns'

            energy_units: (str) Energy units to be used in plotting, available
                                units: 'eV', 'kcal mol-1', 'kJ mol-1'

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable,
                        e.g. [(0.8, 1.5), (80, 120)]

            idx: (int) Integer which specifies which metadynamics run to use
                       for plotting the FES convergence
        """

        logger.info('Plotting FES convergence')

        os.chdir('plumed_files/metadynamics')

        # List of times when a new gaussian is deposited
        deposit_time = np.loadtxt(f'HILLS_{idx}.dat', usecols=0)

        fes_time = [deposit_time[i]
                    for i in range(stride - 1, len(deposit_time), stride)]

        # sum_hills generates surfaces with the stride,
        # but it also always computes the final FES
        remove_duplicate = fes_time[-1] == deposit_time[-1]
        fes_time.append(deposit_time[-1])

        fes_time = convert_ase_time(time_array=np.array(fes_time), units=time_units)
        fes_time = np.round(fes_time, decimals=1)

        self._generate_fes_files(n_bins=n_bins,
                                 cvs_bounds=cvs_bounds,
                                 stride=stride,
                                 idx=idx)

        move_files([fr'fes_{idx}_\d+\.dat'],
                   dst_folder='../fes_convergence',
                   regex=True)
        os.chdir('../fes_convergence')

        # Remove the final FES if it has already been computed with the stride
        # (file enumeration using stride starts from zero)
        if remove_duplicate:
            os.remove(f'fes_{idx}_{len(fes_time)-1}.dat')
            fes_time = fes_time[:-1]

        cv_grids, fes_grids = self._fes_files_to_grids(energy_units=energy_units,
                                                       n_bins=n_bins)

        self._plot_surface_difference(fes_grids=fes_grids,
                                      fes_time=fes_time,
                                      time_units=time_units,
                                      energy_units=energy_units)

        if self.n_cvs == 1:
            self._plot_multiple_1d_fes_surfaces(cv_grids=cv_grids,
                                                fes_grids=fes_grids,
                                                fes_time=fes_time,
                                                n_surfaces=n_surfaces,
                                                time_units=time_units,
                                                energy_units=energy_units)

        os.chdir('../..')

        move_files(['fes_convergence.pdf', 'fes_convergence_diff.pdf'],
                   dst_folder='fes_convergence',
                   src_folder='plumed_files/fes_convergence')

        return None

    @staticmethod
    def _plot_surface_difference(fes_grids: np.ndarray,
                                 fes_time: List,
                                 time_units: str,
                                 energy_units: str
                                 ) -> None:
        """
        Plot the root mean square difference between free energy surfaces
        as a function of time
        """

        fes_diff_grids = np.diff(fes_grids, axis=0)
        rms_diffs = [np.sqrt(np.mean(grid * grid)) for grid in fes_diff_grids]

        fig, ax = plt.subplots()
        ax.plot(fes_time[:-1], rms_diffs)

        ax.set_xlabel(f'Time / {time_units}')
        ax.set_ylabel(r'$\left\langle\Delta\Delta G^{2} '
                      r'\right\rangle^{\frac{1}{2}}$ / '
                      f'{convert_exponents(energy_units)}')

        fig.tight_layout()

        fig.savefig('fes_convergence_diff.pdf')
        plt.close(fig)

        return None

    def _plot_multiple_1d_fes_surfaces(self,
                                       cv_grids: np.ndarray,
                                       fes_grids: np.ndarray,
                                       fes_time: List,
                                       n_surfaces: int,
                                       time_units: str,
                                       energy_units: str
                                       ) -> None:
        """
        Plot multiple 1D free energy surfaces as a function of simulation time
        """

        plotted_cv = self.bias.metad_cvs[0]

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

        ax.set_ylabel(r'$\Delta G$ / '
                      f'{convert_exponents(energy_units)}')

        fig.tight_layout()

        fig.savefig('fes_convergence.pdf')
        plt.close(fig)

    def _generate_fes_files(self,
                            n_bins:      int,
                            cvs_bounds:  Optional[Sequence] = None,
                            stride:      Optional[int] = None,
                            idx:         Optional[int] = None
                            ) -> None:
        """
        Generate fes.dat files from a HILLS.dat file.

        -----------------------------------------------------------------------
        Arguments:

            n_bins: (int) Number of bins to use in every dimension for fes file
                          generation from HILLS

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable

            stride: (int) Interval that specifies the number of new gaussians
                          that is put into subsequent fes_idx_*.dat file

            idx: (int) Integer which specifies which metadynamics run to use
                       for plotting the FES convergence
        """

        if not any(filename.startswith('HILLS') for filename in os.listdir()):
            raise FileNotFoundError('No HILLS.dat files were found in '
                                    'plumed_files, make sure to run '
                                    'metadynamics before computing the FES')

        logger.info('Generating fes.dat files from HILLS.dat files')

        bin_param_seq = ','.join(str(n_bins-1) for _ in range(self.n_cvs))
        min_param_seq, max_param_seq = self._get_min_max_params(cvs_bounds)

        label = '*' if idx is None else idx

        for filename in glob.glob(f'HILLS_{label}.dat'):

            # HILLS_*.dat -> *
            index = filename.split('.')[0].split('_')[-1]

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
                            relative:     bool = True
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use fes.dat files in a current directory to compute a grid containing
        collective variables and a grid containing free energy surfaces.

        -----------------------------------------------------------------------
        Arguments:

            energy_units: (str) Energy units to be used in plotting, available
                                units: 'eV', 'kcal mol-1', 'kJ mol-1'

            n_bins: (int) Number of bins to used in every dimension when fes
                          files were generated

            relative: (bool) If True the energies of the free energy surfaces
                             are shifted such that the global minima are at
                             zero

        Returns:

            (Tuple): Two grids containing CVs and FESs in supplied energy units
        """

        grid_shape = tuple([n_bins for _ in range(self.n_cvs)])

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
            fes_grid = convert_ase_energy(energy_array=fes_grid,
                                          units=energy_units)

            if relative:
                fes_grid -= np.min(fes_grid)

            fes_grids.append(fes_grid)

        total_cv_grid = np.stack(cv_grids, axis=0)
        total_fes_grid = np.stack(fes_grids, axis=0)

        return total_cv_grid, total_fes_grid

    def _get_min_max_params(self,
                            cvs_bounds: Optional[Sequence] = None,
                            path:       Optional[str] = None
                            ) -> Tuple:
        """
        Compute min and max parameters for generating fes.dat files from
        HILLS.dat files.

        -----------------------------------------------------------------------
        Arguments:

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable

            path: (str) Relative path where HILLS.dat files are located

        Returns:

            (Tuple): Two sequences of min and max parameters
        """

        if cvs_bounds is None:
            logger.info('CVs bounds were not supplied, generating min and max '
                        'parameters automatically')

            initial_path = os.getcwd()
            if path is not None:
                path = os.path.join(initial_path, path)
                os.chdir(path)

            min_params, max_params = [], []

            for cv in self.bias.metad_cvs:
                min_values, max_values = [], []

                for filename in glob.glob(f'colvar_{cv.name}_*.dat'):
                    cv_values = np.loadtxt(filename, usecols=1)

                    min_values.append(np.min(cv_values))
                    max_values.append(np.max(cv_values))

                total_min = min(min_values)
                total_max = max(max_values)

                extension = 0.309 * (total_max - total_min)

                min_params.append(str(total_min - extension))
                max_params.append(str(total_max + extension))

            if path is not None:
                os.chdir(initial_path)

            return ','.join(min_params), ','.join(max_params)

        else:
            cvs_bounds_checked = self._check_cv_bounds(cvs_bounds)

            min_params = [str(cv_bounds[0]) for cv_bounds in cvs_bounds_checked]
            max_params = [str(cv_bounds[1]) for cv_bounds in cvs_bounds_checked]

            return ','.join(min_params), ','.join(max_params)

    def _check_cv_bounds(self,
                         cvs_bounds: Sequence
                         ) -> Sequence:
        """
        Check the validity of the supplied CVs bounds and returns the bounds
        in a universal format.

        -----------------------------------------------------------------------
        Arguments:

            cvs_bounds: Specifies the range between which to compute the free
                        energy for each collective variable

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
