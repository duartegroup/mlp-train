import pytest
import numpy as np
import mlptrain as mlt
from mlptrain.box import Box
from mlptrain.configurations import Configuration, ConfigurationSet
from autode.atoms import Atom


@pytest.fixture
def test_system(h2o):
    """Create a sample system with a water molecule and a box."""
    return mlt.System(h2o, box=Box([10, 10, 10]))


@pytest.fixture
def test_charged_system(mg):
    """Create a sample system with a magnesium and a box."""
    return mlt.System(mg, box=Box([10, 10, 10]))


@pytest.fixture
def test_radical_system(oh_radical):
    """Create a sample system with a water molecule and a box."""
    return mlt.System(oh_radical, box=Box([10, 10, 10]))


def test_random_configuration(test_system):
    """Test generating a random configuration for a system."""
    config = test_system.random_configuration(min_dist=1.0)
    assert len(config.atoms) == sum(
        len(m.atoms) for m in test_system.molecules
    )
    assert test_system != config


def test_random_configurations(test_system):
    """Test generating multiple random configurations for a system."""
    configs = test_system.random_configurations(num=5, min_dist=1.0)
    assert isinstance(configs, ConfigurationSet)
    assert len(configs) == 5
    for config in configs:
        assert isinstance(config, Configuration)
        assert config != test_system


def test_add_molecule(test_system, h2o):
    """Test adding a single molecule to a system."""
    initial_count = len(test_system.molecules)
    test_system.add_molecule(h2o)
    assert len(test_system.molecules) == initial_count + 1


def test_add_multiple_molecules(test_system, h2o):
    """Test adding multiple molecules to a system."""
    initial_count = len(test_system.molecules)
    test_system.add_molecules(h2o, num=3)
    assert len(test_system.molecules) == initial_count + 3


def test_charge_property(test_charged_system, mg):
    """Test the system's total charge property."""
    assert test_charged_system.charge == sum(
        m.charge for m in test_charged_system.molecules
    )


def test_mult_property(test_radical_system, oh_radical):
    """Test the system's total multiplicity property."""
    expected_mult = 2
    assert test_radical_system.mult == expected_mult


def test_atoms_property(test_system):
    """Test getting all atoms in the system."""
    atoms = test_system.atoms
    assert isinstance(atoms, list)
    assert all(isinstance(atom, Atom) for atom in atoms)
    total_atoms = sum(len(m.atoms) for m in test_system.molecules)
    assert len(atoms) == total_atoms


def test_unique_atomic_symbols_property(test_system):
    """Test getting unique atomic symbols in the system."""
    unique_symbols = test_system.unique_atomic_symbols
    expected_symbols = set(
        atom.label for mol in test_system.molecules for atom in mol.atoms
    )
    assert set(unique_symbols) == expected_symbols


def test_shift_randomly_raises_runtimeerror_on_failure(test_system, h2o):
    """Test that _shift_randomly raises RuntimeError after max attempts."""
    with pytest.raises(RuntimeError):
        test_system._shift_randomly(
            h2o, coords=np.array([[0, 0, 0]]), min_dist=100
        )
