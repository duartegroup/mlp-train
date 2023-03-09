import os
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


def _run_metadynamics(metadynamics,
                      n_runs,
                      save_sep=False,
                      all_to_xyz=False,
                      restart=False,
                      **kwargs):
    metadynamics.run_metadynamics(configuration=_h2_configuration(),
                                  mlp=TestPotential('1D'),
                                  temp=300,
                                  interval=10,
                                  dt=1,
                                  pace=100,
                                  width=0.05,
                                  height=0.1,
                                  biasfactor=3,
                                  n_runs=n_runs,
                                  save_sep=save_sep,
                                  all_to_xyz=all_to_xyz,
                                  restart=restart,
                                  ps=1,
                                  **kwargs)


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    assert metad.bias is not None

    _run_metadynamics(metad, n_runs, all_to_xyz=True, save_fs=300)

    assert os.path.exists('trajectories')
    assert os.path.exists('trajectories/combined_trajectory.xyz')

    metad_dir = 'plumed_files/metadynamics'
    for idx in range(1, n_runs + 1):
        assert os.path.exists(f'trajectories/trajectory_{idx}.traj')

        for sim_time in [300, 600, 900]:
            assert os.path.exists(f'trajectories/trajectory_{idx}_{sim_time}fs.traj')
            assert os.path.exists(f'trajectories/metad_{idx}_{sim_time}fs.xyz')

        assert os.path.exists(os.path.join(metad_dir, f'colvar_cv1_{idx}.dat'))
        assert os.path.exists(os.path.join(metad_dir, f'HILLS_{idx}.dat'))

    metad.compute_fes(n_bins=100)

    for idx in range(1, n_runs + 1):
        assert os.path.exists(f'plumed_files/metadynamics/fes_{idx}.dat')

    assert os.path.exists('fes_raw.npy')
    fes_raw = np.load('fes_raw.npy')

    # 1 cv, 4 fes -> 5; 100 bins + 1 -> 101
    assert np.shape(fes_raw) == (5, 101)

    metad.plot_fes('fes_raw.npy')
    assert os.path.exists('metad_free_energy.pdf')

    metad.plot_fes_convergence(stride=2, n_surfaces=3)

    # 1000 / 100: simulation time divided by the pace <=> number of gaussians
    # Surfaces are computed every 2 gaussians
    n_computed_surfaces = (1000 / 100) // 2
    for idx in range(int(n_computed_surfaces)):
        assert os.path.exists(f'plumed_files/fes_convergence/fes_1_{idx}.dat')

    assert os.path.exists('fes_convergence/fes_convergence_diff.pdf')
    assert os.path.exists('fes_convergence/fes_convergence.pdf')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics_restart():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    _run_metadynamics(metad, n_runs)

    _run_metadynamics(metad, n_runs, restart=True)

    n_steps = len(np.loadtxt('plumed_files/metadynamics/colvar_cv1_1.dat',
                             usecols=0))
    n_gaussians = len(np.loadtxt('plumed_files/metadynamics/HILLS_1.dat',
                                 usecols=0))

    # Adding two 1 ps simulations with interval 10 -> 101 frames each, but
    # removing one duplicate frame
    assert n_steps == 101 + 101 - 1
    assert n_gaussians == 10 + 10

    assert os.path.exists('trajectories/trajectory_1.traj')

    trajectory = ASETrajectory('trajectories/trajectory_1.traj')

    # Adding two 1 ps simulations with interval 10 -> 101 frames each, but
    # removing one duplicate frame (same as before, except testing this for
    # the generated .traj file instead of .dat file)
    assert len(trajectory) == 101 + 101 - 1


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics_with_component():

    cv1 = mlt.PlumedCustomCV('plumed_cv_dist.dat', 'x')
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    _run_metadynamics(metad, n_runs)

    metad_dir = 'plumed_files/metadynamics'
    for idx in range(1, n_runs + 1):
        assert os.path.exists(os.path.join(metad_dir, f'colvar_cv1_x_{idx}.dat'))


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
    biasfactors = range(5, 16, 5)

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

    plots_dir = 'multiple_biasfactors'
    assert os.path.isdir(plots_dir)

    for idx, biasf in enumerate(biasfactors, start=1):
        assert os.path.exists(os.path.join(files_dir, f'colvar_cv1_{idx}.dat'))
        assert os.path.exists(os.path.join(plots_dir, f'cv1_biasf{biasf}.pdf'))

@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_block_analysis():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 1

    _run_metadynamics(metad, n_runs)

    metad.block_analysis()

    assert os.path.exists('block_analysis.pdf')
    assert os.path.exists('block_analysis_error.npy')
