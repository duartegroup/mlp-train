from autode.atoms import Atom
from mlptrain.configurations.configuration import Configuration
from mlptrain.configurations.trajectory import Trajectory


def test_trajectory_allows_duplicates():

    traj = Trajectory(Configuration(atoms=[Atom("H")]))
    traj.append(Configuration(atoms=[Atom("H")]))
    assert len(traj) == 2
