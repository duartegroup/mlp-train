import os
import mlptrain as mlt
from ase.io.trajectory import Trajectory as ASETrajectory
from .test_potential import TestPotential
from .molecules import _h2
from .utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


def _h2_configuration():
    system = mlt.System(_h2(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_md_full_plumed_input():

    bias = mlt.PlumedBias(file_name='plumed_bias.dat')

    mlt.md.run_mlp_md(configuration=_h2_configuration(),
                      mlp=TestPotential('1D'),
                      temp=300,
                      dt=1,
                      interval=10,
                      bias=bias,
                      kept_substrings=['.dat'],
                      ps=1)

    assert os.path.exists('colvar.dat')
    assert os.path.exists('HILLS.dat')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_md_restart():

    atoms = _h2_configuration().ase_atoms
    initial_trajectory = ASETrajectory('md_restart.traj', 'r', atoms)

    mlt.md.run_mlp_md(configuration=_h2_configuration(),
                      mlp=TestPotential('1D'),
                      temp=300,
                      dt=1,
                      interval=10,
                      restart_files=['md_restart.traj'],
                      ps=1)

    assert os.path.exists('md_restart.traj')

    final_trajectory = ASETrajectory('md_restart.traj', 'r', atoms)

    # 10 ps simulation with interval 10 -> 1001 frames
    assert len(initial_trajectory) == 1001

    # Adding 1 ps simulation with interval 10 -> 101 frames, but removing one
    # duplicate frame
    assert len(final_trajectory) == 1001 + 101 - 1


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_md_save():

    mlt.md.run_mlp_md(configuration=_h2_configuration(),
                      mlp=TestPotential('1D'),
                      temp=300,
                      dt=1,
                      interval=10,
                      kept_substrings=['.traj'],
                      ps=1,
                      save_fs=200)

    assert os.path.exists('trajectory.traj')

    assert not os.path.exists('trajectory_0fs.traj')
    assert os.path.exists('trajectory_200fs.traj')
    assert os.path.exists('trajectory_1000fs.traj')
    assert not os.path.exists('trajectory_1200fs.traj')

    traj_200fs = ASETrajectory('trajectory_200fs.traj')

    # 200 ps / 10 interval == 20 frames; + 1 starting frame
    assert len(traj_200fs) == 20 + 1
