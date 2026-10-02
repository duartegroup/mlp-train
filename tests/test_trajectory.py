import logging
import pickle

import pytest
from autode.atoms import Atom

from mlptrain.configurations.configuration import Configuration
from mlptrain.configurations.trajectory import Trajectory


def _frame(time=None, x=0.0):
    config = Configuration(atoms=[Atom('H', x, 0.0, 0.0)])
    config.time = time
    return config


def test_trajectory_allows_duplicates():
    config = _frame()
    traj = Trajectory(config, config, _frame())
    traj.append(config)
    assert len(traj) == 4


def test_trajectory_is_pickleable():
    traj = Trajectory(_frame(time=1.0, x=0.0), _frame(time=2.0, x=1.0))
    traj.append(traj[0])

    unpickled = pickle.loads(pickle.dumps(traj))

    assert isinstance(unpickled, Trajectory)
    assert unpickled.allow_duplicates
    assert len(unpickled) == 3
    assert [frame.time for frame in unpickled] == [1.0, 2.0, 1.0]
    assert unpickled.t0 == 1.0
    assert unpickled.final_frame.coordinates[0][0] == 0.0


def test_t0_empty_trajectory():
    assert Trajectory().t0 == 0.0


def test_t0_is_time_of_first_frame():
    traj = Trajectory(_frame(time=2.0), _frame(time=3.0))
    assert traj.t0 == 2.0


def test_t0_undefined_time():
    traj = Trajectory(_frame())
    assert traj.t0 is None


def test_t0_setter_shifts_defined_times():
    traj = Trajectory(_frame(time=0.0), _frame(time=1.0), _frame(time=2.5))

    traj.t0 = 10.0

    assert [frame.time for frame in traj] == [10.0, 11.0, 12.5]
    assert traj.t0 == 10.0


def test_t0_setter_undefined_times(mlp_caplog):
    mlp_caplog.set_level(logging.WARNING, logger='mlptrain')
    traj = Trajectory(_frame(), _frame(time=1.0))

    traj.t0 = 5.0

    assert [frame.time for frame in traj] == [5.0, 6.0]
    assert 'Setting to 5.0' in mlp_caplog.text


def test_t0_setter_empty_trajectory():
    traj = Trajectory()
    traj.t0 = 5.0
    assert len(traj) == 0
    assert traj.t0 == 0.0


def test_final_frame():
    first, last = _frame(x=0.0), _frame(x=1.0)
    traj = Trajectory(first, last)

    assert traj.final_frame is last


def test_final_frame_single_frame():
    frame = _frame()
    assert Trajectory(frame).final_frame is frame


def test_final_frame_empty_trajectory():
    with pytest.raises(ValueError, match='no final frame'):
        _ = Trajectory().final_frame
