import os
import glob
import numpy as np
import mlptrain as mlt
from ase.io.trajectory import Trajectory as ASETrajectory
from .test_potential import TestPotential
from .molecules import _h2
from .utils import work_in_zipped_dir
mlt.Config.n_cores = 2
here = os.path.abspath(os.path.dirname(__file__))


def _h2_configuration():
    system = mlt.System(_h2(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


def _run_metadynamics(metadynamics, restart=False):
    metadynamics.run_metadynamics(configuration=_h2_configuration(),
                                  mlp=TestPotential('1D'),
                                  temp=300,
                                  interval=10,
                                  dt=1,
                                  pace=100,
                                  width=0.05,
                                  height=0.1,
                                  biasfactor=3,
                                  n_runs=4,
                                  ps=1,
                                  save_sep=True,
                                  restart=restart)


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    assert metad.bias is not None

    _run_metadynamics(metad)

    assert os.path.exists('trajectories')
    assert glob.glob('trajectories/trajectory_*.traj')
    assert glob.glob('trajectories/metad_*.xyz')

    metad_dir = 'plumed_files/metadynamics'
    assert glob.glob(os.path.join(metad_dir, 'colvar_cv1_*.dat'))
    assert glob.glob(os.path.join(metad_dir, 'HILLS_*.dat'))

    metad.compute_fes(n_bins=100)

    assert glob.glob('plumed_files/metadynamics/fes_*.dat')

    assert os.path.exists('fes_raw.npy')

    fes_raw = np.load('fes_raw.npy')

    # 1 cv, 4 fes -> 5; 100 bins + 1 -> 101
    assert np.shape(fes_raw) == (5, 101)

    metad.plot_fes('fes_raw.npy')

    assert os.path.exists('metad_free_energy.pdf')

    metad.plot_fes_convergence(stride=2, n_surfaces=3)

    assert glob.glob('plumed_files/fes_convergence/fes_1_*.dat')

    assert os.path.exists('fes_convergence/fes_convergence_diff.pdf')
    assert os.path.exists('fes_convergence/fes_convergence.pdf')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics_restart():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    _run_metadynamics(metad)

    _run_metadynamics(metad, restart=True)

    n_steps = len(np.loadtxt('plumed_files/metadynamics/colvar_cv1_1.dat',
                             usecols=0))
    n_gaussians = len(np.loadtxt('plumed_files/metadynamics/HILLS_1.dat',
                                 usecols=0))

    assert n_steps == 101 + 101 - 1
    assert n_gaussians == 10 + 10

    assert os.path.exists('trajectories/trajectory_1.traj')

    trajectory = ASETrajectory('trajectories/trajectory_1.traj')

    assert len(trajectory) == 101 + 101 - 1


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics_with_component():

    cv1 = mlt.PlumedCustomCV('plumed_cv_dist.dat', 'x')
    metad = mlt.Metadynamics(cv1)

    _run_metadynamics(metad)

    metad_dir = 'plumed_files/metadynamics'
    assert glob.glob(os.path.join(metad_dir, 'colvar_cv1_x_*.dat'))


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_estimate_width():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    width = metad.estimate_width(configurations=_h2_configuration(),
                                 mlp=TestPotential('1D'),
                                 plot=True,
                                 fs=500)

    assert len(width) == 1

    files_directory = 'plumed_files/width_estimation'
    plots_directory = 'width_estimation'

    assert os.path.isdir(files_directory)
    assert os.path.exists(os.path.join(files_directory, 'colvar_cv1_1.dat'))

    assert os.path.isdir(plots_directory)
    assert os.path.exists(os.path.join(plots_directory, 'cv1_config1.pdf'))


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_try_multiple_biasfactors():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    metad.try_multiple_biasfactors(configuration=_h2_configuration(),
                                   mlp=TestPotential('1D'),
                                   temp=300,
                                   interval=10,
                                   dt=1,
                                   pace=100,
                                   width=0.05,
                                   height=0.1,
                                   biasfactors=range(5, 16, 5),
                                   plotted_cvs=cv1,
                                   ps=1)

    files_dir = 'plumed_files/multiple_biasfactors'
    assert os.path.isdir(files_dir)
    assert glob.glob(os.path.join(files_dir, 'colvar_cv1_*.dat'))

    plots_dir = 'multiple_biasfactors'
    assert os.path.isdir(plots_dir)
    assert glob.glob(os.path.join(plots_dir, 'cv1_biasf*.pdf'))
