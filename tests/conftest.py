import mlptrain as mlt
import pytest
import numpy as np
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
    """Create a water molecule (H2O) for testing."""
    # Approximate coordinates for H2O with a 104.5° bond angle and 0.96 Å bond length
    oxygen = Atom('O', 0.0, 0.0, 0.0)
    hydrogen1 = Atom('H', 0.96, 0.0, 0.0)
    hydrogen2 = Atom(
        'H',
        -0.96 * np.cos(np.radians(104.5)),
        0.96 * np.sin(np.radians(104.5)),
        0.0,
    )
    return mlt.Molecule(atoms=[oxygen, hydrogen1, hydrogen2], charge=0, mult=1)
