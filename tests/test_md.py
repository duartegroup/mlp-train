import logging
import os
import signal
import time

import numpy as np
import pytest
import mlptrain as mlt
from ase.io.trajectory import Trajectory as ASETrajectory
from ase.constraints import Hookean

from .data.utils import work_in_zipped_dir

here = os.path.abspath(os.path.dirname(__file__))


def test_run_with_timeout_returns_result_when_fast():
    result, finished_in_time = mlt.md.run_with_timeout(
        lambda x: x + 1, 1, fn_timeout=30
    )

    assert (result, finished_in_time) == (2, True)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_run_with_timeout_cancels_slow_call():
    """A call that overruns is abandoned rather than left to run forever."""

    original_handler = signal.getsignal(signal.SIGALRM)

    result, finished_in_time = mlt.md.run_with_timeout(
        time.sleep, 30, fn_timeout=0.1
    )

    assert (result, finished_in_time) == (None, False)

    # The timer must be cancelled and the previous handler restored, otherwise
    # a later unrelated call in the same worker inherits a live alarm
    assert signal.getsignal(signal.SIGALRM) is original_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_run_with_timeout_propagates_other_exceptions():
    """Exception raised in `fn` must be propagated to the caller."""

    n_calls = []

    def _boom():
        n_calls.append(1)
        raise ValueError('bad forces')

    original_handler = signal.getsignal(signal.SIGALRM)

    with pytest.raises(ValueError, match='bad forces'):
        mlt.md.run_with_timeout(_boom, fn_timeout=30)

    assert len(n_calls) == 1
    assert signal.getsignal(signal.SIGALRM) is original_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_run_with_timeout_falls_back_when_alarm_unavailable(monkeypatch):
    """Without a usable SIGALRM the call still runs, just unbounded."""

    def _no_alarm(*args, **kwargs):
        raise ValueError('SIGALRM not available here')

    monkeypatch.setattr(signal, 'setitimer', _no_alarm)
    original_handler = signal.getsignal(signal.SIGALRM)

    result, finished_in_time = mlt.md.run_with_timeout(
        lambda: 'ok', fn_timeout=0.1
    )

    assert (result, finished_in_time) == ('ok', True)
    assert signal.getsignal(signal.SIGALRM) is original_handler


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_md_full_plumed_input(h2o_configuration, test_potential):
    bias = mlt.PlumedBias(filename='plumed_bias_nopath.dat')

    mlt.md.run_mlp_md(
        configuration=h2o_configuration,
        mlp=test_potential('1D'),
        temp=300,
        dt=1,
        interval=10,
        bias=bias,
        kept_substrings=['.dat'],
        ps=1,
    )

    assert os.path.exists('colvar.dat')
    assert os.path.exists('HILLS.dat')


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_md_restart(h2_configuration, test_potential):
    atoms = h2_configuration.ase_atoms
    initial_trajectory = ASETrajectory('md_restart.traj', 'r', atoms)

    mlt.md.run_mlp_md(
        configuration=h2_configuration,
        mlp=test_potential('1D'),
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


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_md_save(h2_configuration, test_potential):
    mlt.md.run_mlp_md(
        configuration=h2_configuration,
        mlp=test_potential('1D'),
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


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_md_traj_attachments(h2o_configuration, test_potential):
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    bias = mlt.PlumedBias(cvs=cv1)

    hookean_constraint = Hookean(a1=1, a2=2, k=100, rt=0.5)

    traj = mlt.md.run_mlp_md(
        configuration=h2o_configuration,
        mlp=test_potential('1D'),
        temp=300,
        dt=1,
        interval=10,
        bias=bias,
        kept_substrings=['colvar_cv1.dat'],
        constraints=[hookean_constraint],
        ps=1,
    )

    assert traj is not None

    plumed_coordinates = np.loadtxt('colvar_cv1.dat', usecols=1)

    for i, config in enumerate(traj):
        assert np.shape(config.plumed_coordinates) == (1,)
        assert config.plumed_coordinates[0] == plumed_coordinates[i]

    assert all(bias_energy is not None for bias_energy in traj.bias_energies)
    assert any(bias_energy != 0 for bias_energy in traj.bias_energies)


def test_sim_time_print(mlp_caplog):
    from mlptrain.sampling.md import _log_sim_time

    mlp_caplog.set_level(logging.INFO, logger='mlptrain')

    _log_sim_time(1.2501)
    assert len(mlp_caplog.records) == 1
    assert (
        mlp_caplog.records[0].message
        == 'MLP MD simulation completed in 1.25 s.'
    )

    _log_sim_time(61.0)
    assert len(mlp_caplog.records) == 2
    assert (
        mlp_caplog.records[1].message
        == 'MLP MD simulation completed in 00 h 01 min 1.00 s.'
    )

    _log_sim_time(3600 + 60 + 5.7)
    assert len(mlp_caplog.records) == 3
    assert (
        mlp_caplog.records[2].message
        == 'MLP MD simulation completed in 01 h 01 min 5.70 s.'
    )

    _log_sim_time(24 * 3600 + 12 * 3600 + 7 * 60 + 0.01)
    assert len(mlp_caplog.records) == 4
    assert (
        mlp_caplog.records[3].message
        == 'MLP MD simulation completed in 1 day 12 h 07 min 0.01 s.'
    )

    _log_sim_time(4 * 24 * 3600 + 3 * 3600 + 11 * 60 + 0)
    assert len(mlp_caplog.records) == 5
    assert (
        mlp_caplog.records[4].message
        == 'MLP MD simulation completed in 4 days 03 h 11 min 0.00 s.'
    )
