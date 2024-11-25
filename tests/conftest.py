import mlptrain as mlt
import pytest
from autode.atoms import Atom


@pytest.fixture
def h2():
    """Dihydrogen molecule"""
    atoms = [
        Atom('H', -0.80952, 2.49855, 0.0),
        Atom('H', -0.34877, 1.961, 0.0),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)


@pytest.fixture
def h2_configuration(h2):
    system = mlt.System(h2, box=[50, 50, 50])
    config = system.random_configuration()

    return config


@pytest.fixture
def h2o_configuration(h2o):
    system = mlt.System(h2o, box=[50, 50, 50])
    config = system.random_configuration()

    return config


@pytest.fixture
def h2o():
    """Water molecule"""
    atoms = [
        Atom('H', 2.32670, 0.51322, 0.0),
        Atom('H', 1.03337, 0.70894, -0.89333),
        Atom('O', 1.35670, 0.51322, 0.0),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)
