import numpy as np
import pytest
from ase.build import molecule
from mlptrain.descriptor import SoapDescriptor
from mlptrain import Configuration, ConfigurationSet


# Fixtures for configurations
@pytest.fixture
def simple_molecule():
    from ase.build import molecule

    return Configuration(ase_atoms=molecule('H2O'))


@pytest.fixture
def configuration_set(simple_molecule):
    # Create a configuration set with duplicates of the simple molecule
    return ConfigurationSet([simple_molecule, simple_molecule])


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
    assert representation.shape == (
        1,
        descriptor.n_max * descriptor.l_max * len(descriptor.elements) ** 2,
    )


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
    # Different molecules: water and methane
    water = Configuration(ase_atoms=molecule('H2O'))
    methane = Configuration(ase_atoms=molecule('CH4'))
    config_set = ConfigurationSet([water, methane])

    descriptor = SoapDescriptor(
        elements=['H', 'C', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    kernel_vector = descriptor.kernel_vector(water, config_set, zeta=4)
    # Check values are reasonable; they should not be 1 since molecules differ
    assert not np.allclose(kernel_vector, np.ones(len(config_set)), atol=1e-5)

