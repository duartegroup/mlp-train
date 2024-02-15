import os
import numpy as np
import mlptrain as mlt
from ase.io.trajectory import Trajectory as ASETrajectory
from ase.constraints import Hookean
from .test_potential import TestPotential
from .molecules import _h2, _h2o
from .utils import work_in_zipped_dir

here = os.path.abspath(os.path.dirname(__file__))


def _h2_configuration():
    system = mlt.System(_h2(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


def _h2o_configuration():
    system = mlt.System(_h2o(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_md_full_plumed_input():
    bias = mlt.PlumedBias(filename='plumed_bias_nopath.dat')

    mlt.md.run_mlp_md(
        configuration=_h2o_configuration(),
        mlp=TestPotential('1D'),
        temp=300,
        dt=1,
        interval=10,
        bias=bias,
        kept_substrings=['.dat'],
        ps=1,
    )

    assert os.path.exists('colvar.dat')
    assert os.path.exists('HILLS.dat')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_md_restart():
    atoms = _h2_configuration().ase_atoms
    initial_trajectory = ASETrajectory('md_restart.traj', 'r', atoms)

    mlt.md.run_mlp_md(
        configuration=_h2_configuration(),
        mlp=TestPotential('1D'),
        temp=300,
        dt=1,
        interval=10,
        restart_files=['md_restart.traj'],
        ps=1,
    )

    assert os.path.exists('md_restart.traj')

    final_trajectory = ASETrajectory('md_restart.traj', 'r', atoms)

    # 10 ps simulation with dt = 1 fs and interval of 10 -> 1001 frames
    assert len(initial_trajectory) == 1001

    # Adding 1 ps simulation with interval 10 -> 101 frames, but removing one
    # duplicate frame
    assert len(final_trajectory) == 1001 + 101 - 1


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_md_save():
    mlt.md.run_mlp_md(
        configuration=_h2_configuration(),
        mlp=TestPotential('1D'),
        temp=300,
        dt=1,
        interval=10,
        kept_substrings=['.traj'],
        ps=1,
        save_fs=200,
    )

    assert os.path.exists('trajectory.traj')

    assert not os.path.exists('trajectory_0fs.traj')
    assert os.path.exists('trajectory_200fs.traj')
    assert os.path.exists('trajectory_1000fs.traj')
    assert not os.path.exists('trajectory_1200fs.traj')

    traj_200fs = ASETrajectory('trajectory_200fs.traj')

    # 200 ps / 10 interval == 20 frames; + 1 starting frame
    assert len(traj_200fs) == 20 + 1


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_md_traj_attachments():
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    bias = mlt.PlumedBias(cvs=cv1)

    hookean_constraint = Hookean(a1=1, a2=2, k=100, rt=0.5)

    traj = mlt.md.run_mlp_md(
        configuration=_h2o_configuration(),
        mlp=TestPotential('1D'),
        temp=300,
        dt=1,
        interval=10,
        bias=bias,
        kept_substrings=['colvar_cv1.dat'],
        constraints=[hookean_constraint],
        ps=1,
    )

    plumed_coordinates = np.loadtxt('colvar_cv1.dat', usecols=1)

    for i, config in enumerate(traj):
        assert np.shape(config.plumed_coordinates) == (1,)
        assert config.plumed_coordinates[0] == plumed_coordinates[i]

    assert all(bias_energy is not None for bias_energy in traj.bias_energies)
    assert any(bias_energy != 0 for bias_energy in traj.bias_energies)
