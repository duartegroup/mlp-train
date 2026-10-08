import logging
import os

import autode
import autode.values
import numpy as np
import pytest
from ase.calculators.lj import Calculator, LennardJones
from autode.atoms import Atom

import mlptrain as mlt
from mlptrain.log import logger as mlp_logger


@pytest.fixture(autouse=True, scope='session')
def patch_autode_config():
    """Patch autode's default memory requirements to make sure
    that tests can run on hardware with less than 16Gb RAM
    """
    autode.config.Config.max_core = autode.values.Allocation(1, units='GB')


@pytest.fixture
def mlp_caplog(caplog):
    """``caplog`` wired up to the mlptrain logger.

    The project logger sets ``propagate = False`` to avoid duplicating
    records into root handlers. As a side effect pytest's ``caplog`` handler
    never sees mlptrain records unless it is attached to that logger
    directly, so a bare ``caplog`` silently reports zero records.
    """
    caplog.set_level(logging.INFO, logger='mlptrain')
    mlp_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        mlp_logger.removeHandler(caplog.handler)


@pytest.fixture
def h2():
    """Dihydrogen molecule"""
    atoms = [
        Atom('H', -0.80952, 2.49855, 0.0),
        Atom('H', -0.34877, 1.961, 0.0),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)


@pytest.fixture
def h2o():
    """Water molecule"""
    atoms = [
        Atom('H', 2.32670, 0.51322, 0.0),
        Atom('H', 1.03337, 0.70894, -0.89333),
        Atom('O', 1.35670, 0.51322, 0.0),
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
def h2o_configuration_set(h2o):
    system = mlt.System(h2o, box=[50, 50, 50])
    config_set = mlt.ConfigurationSet()
    config_set.append(system.random_configuration())
    config_set.append(system.random_configuration())

    return config_set


@pytest.fixture
def mg():
    "Magnesium cation 2+"
    atoms = [
        Atom('Mg', 0.0, 0.0, 0.0),
    ]
    return mlt.Molecule(atoms=atoms, charge=+2, mult=1)


@pytest.fixture
def oh_radical():
    "OH radiacal species"
    atoms = [
        Atom('O', 1.35670, 0.51322, 0.0),
        Atom('H', 2.32670, 0.51322, 0.0),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=2)


@pytest.fixture
def empty_molecule():
    "No molecule inserted"
    molecule = mlt.Molecule()
    return molecule


@pytest.fixture
def chdir_tmp_path(request, tmp_path):
    """Change to a temporary directory before running the test and reverting to original working directory."""
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(request.config.invocation_dir)


class HarmonicPotential(Calculator):
    __test__ = False

    def get_potential_energy(self, atoms):  # ty:ignore[invalid-method-override]
        r = atoms.get_distance(0, 1)

        return (r - 1) ** 2

    def get_forces(self, atoms):  # ty:ignore[invalid-method-override]
        derivative = np.zeros((len(atoms), 3))

        r = atoms.get_distance(0, 1)

        x_dist, y_dist, z_dist = [
            atoms[0].position[j] - atoms[1].position[j] for j in range(3)
        ]

        x_i, y_i, z_i = (x_dist / r), (y_dist / r), (z_dist / r)

        derivative[0] = [x_i, y_i, z_i]
        derivative[1] = [-x_i, -y_i, -z_i]

        force = -2 * derivative * (r - 1)

        return force


class TestPotential(mlt.potentials.MLPotential):
    __test__ = False

    def __init__(self, name: str, system, calculator='harmonic'):
        super().__init__(name=name, system=system)
        self.calculator = calculator.lower()

    @property
    def ase_calculator(self):
        if self.calculator == 'harmonic':
            return HarmonicPotential()

        if self.calculator == 'lj':
            return LennardJones(rc=2.5, r0=3.0)

        else:
            raise NotImplementedError(
                f'{self.calculator} is not implemented as a test potential'
            )

    def _train(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""

    def requires_atomic_energies(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""

    def requires_non_zero_box_size(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""


@pytest.fixture
def test_potential():
    """Dummy MLPotential"""

    def _create_potential(
        name: str = 'test', calculator: str = 'harmonic', system=None
    ):
        return TestPotential(name, system, calculator=calculator)

    return _create_potential
