import numpy as np
import pytest
from autode.atoms import Atom
from mlptrain.descriptor import SoapDescriptor
from mlptrain import Configuration, ConfigurationSet


def water():
    """Function to create a water molecule Configuration."""
    atoms = [
        Atom('O', 0.0, 0.0, 0.0),
        Atom('H', 0.9572, 0.0, 0.0),
        Atom('H', -0.239006, 0.926627, 0.0),
    ]
    return Configuration(atoms=atoms)


# Fixtures for configurations
@pytest.fixture
def simple_molecule():
    """Fixture to return a simple water molecule Configuration."""
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
    """Test kernel vector calculation where both molecules are identical."""
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    kernel_vector = descriptor.kernel_vector(
        configuration_set[0], configuration_set, zeta=4
    )
    # Kernel between identical molecules should be close to 1 due to normalization
    np.testing.assert_allclose(
        kernel_vector, np.ones(len(configuration_set)), atol=1e-5
    )


def test_kernel_vector_different_molecules():
    """Test kernel vector calculation with different molecules."""
    water_instance = (
        water()
    )  # Call the function to get a Configuration instance
    methane = Configuration(
        atoms=[  # Define methane similarly
            Atom('C', 0, 0, 0),
            Atom('H', 1, 0, 0),
            Atom('H', -1, 0, 0),
            Atom('H', 0, 1, 0),
            Atom('H', 0, -1, 0),
        ]
    )
    config_set = ConfigurationSet(water_instance, methane)  # Proper unpacking

    descriptor = SoapDescriptor(
        elements=['H', 'C', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    kernel_vector = descriptor.kernel_vector(water, config_set, zeta=4)
    # Check values are reasonable; they should not be 1 since molecules differ
    assert not np.allclose(kernel_vector, np.ones(len(config_set)), atol=1e-5)
