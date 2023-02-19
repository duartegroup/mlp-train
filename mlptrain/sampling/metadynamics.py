import os
import time
from typing import Optional, Sequence, Union
from multiprocessing import Pool
from mlptrain.configurations import ConfigurationSet
from mlptrain.sampling.md import run_mlp_md
from mlptrain.sampling.plumed import PlumedBias
from mlptrain.utils import move_files, unique_dirname
from mlptrain.config import Config
from mlptrain.log import logger


def _run_single_metadynamics(start_config, mlp, temp, interval, dt, bias,
                             **kwargs):
    """Initiates a single well-tempered metadynamics run"""

    traj = run_mlp_md(configuration=start_config,
                      mlp=mlp,
                      temp=temp,
                      dt=dt,
                      interval=interval,
                      bias=bias,
                      method='metadynamics',
                      **kwargs)

    return traj


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

    def run_metadynamics(self,
                         start_config: 'mlptrain.Configuration',
                         mlp: 'mlptrain.potentials._base.MLPotential',
                         temp: float,
                         interval: int,
                         dt: float,
                         pace: int,
                         width: float,
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

            for idx in range(n_processes):

                logger.info(f'Running Metadynamics simulation number {idx+1}')

                metad_process = pool.apply_async(func=_run_single_metadynamics,
                                                 args=(start_config,
                                                       mlp,
                                                       temp,
                                                       interval,
                                                       dt,
                                                       self.bias),
                                                 kwds=kwargs)
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

        move_files('.dat', 'plumed_files')
        move_files('.log', 'plumed_logs')

        finish_metadynamics = time.perf_counter()
        logger.info('Metadynamics done in '
                    f'{(finish_metadynamics - start_metadynamics) / 60:.1f} m')

        return None
