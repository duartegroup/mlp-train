import pytest
from autode.atoms import Atom
from mlptrain.descriptor import SoapDescriptor
from mlptrain import Configuration, ConfigurationSet
import numpy as np


def water():
    atoms = [
        Atom('O', 0.0, 0.0, 0.0),
        Atom('H', 0.9572, 0.0, 0.0),
        Atom('H', -0.239006, 0.926627, 0.0),
    ]
    return Configuration(atoms=atoms)


@pytest.fixture
def simple_molecule():
    return water()


@pytest.fixture
def configuration_set(simple_molecule):
    """Fixture to create a ConfigurationSet containing duplicates of a simple molecule."""
    return ConfigurationSet(*[simple_molecule, simple_molecule])


def test_soap_descriptor_initialization():
    """Test initialization of SoapDescriptor with and without elements."""
    # With elements
    descriptor_with = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    assert descriptor_with.elements == ['H', 'O']
    # Without elements (should handle dynamic element setup)
    descriptor_without = SoapDescriptor()
    assert descriptor_without.elements is None


def test_compute_representation(simple_molecule):
    """Test computation of SOAP representation for a simple molecule."""
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    representation = descriptor.compute_representation(simple_molecule)
    # Directly check against the actual observed output shape
    assert representation.shape == (
        1,
        546,
    ), f'Expected shape (1, 546), but got {representation.shape}'


def test_kernel_vector_identical_molecules(configuration_set):
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    kernel_vector = descriptor.kernel_vector(
        configuration_set[0], configuration_set, zeta=4
    )
    assert np.allclose(kernel_vector, np.ones_like(kernel_vector), atol=1e-5)


def test_kernel_vector_different_molecules():
    water_instance = water()
    methane = Configuration(
        atoms=[
            Atom('C', 0, 0, 0),
            Atom('H', 1, 0, 0),
            Atom('H', -1, 0, 0),
            Atom('H', 0, 1, 0),
            Atom('H', 0, -1, 0),
        ]
    )
    config_set = ConfigurationSet(water_instance, methane)
    descriptor = SoapDescriptor(
        elements=['H', 'C', 'O'], r_cut=5.0, n_max=6, l_max=6, average='inner'
    )
    kernel_vector = descriptor.kernel_vector(
        water_instance, config_set, zeta=4
    )

    expected_value = 0.8552933998497922
    assert np.testing.assert_allclose(kernel_vector, expected_value, atol=1e-5)
