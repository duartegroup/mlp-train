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

    assert len(initial_trajectory) == 1001
    assert len(final_trajectory) == 1001 + 101 - 1
