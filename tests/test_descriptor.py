import pytest
from autode.atoms import Atom
from mlptrain.descriptor import SoapDescriptor
from mlptrain import Configuration, ConfigurationSet
import numpy as np


@pytest.fixture
def methane():
    """Fixture to create a Configuration instance for methane."""
    atoms = [
        Atom('C', 0, 0, 0),
        Atom('H', 0.629118, 0.629118, 0.629118),
        Atom('H', -0.629118, -0.629118, 0.629118),
        Atom('H', 0.629118, -0.629118, -0.629118),
        Atom('H', -0.629118, 0.629118, -0.629118),
    ]
    return Configuration(atoms=atoms)


@pytest.fixture
def same_configuration_set(h2o_configuration):
    """Fixture to create a ConfigurationSet containing duplicates of a simple molecule."""
    return ConfigurationSet(h2o_configuration, h2o_configuration)


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


def test_compute_representation(h2o_configuration):
    """Test computation of SOAP representation for water"""
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    representation = descriptor.compute_representation(h2o_configuration)
    # Directly check against the actual observed output shape
    assert representation.shape == (
        1,
        546,
    ), f'Expected shape (1, 546), but got {representation.shape}'


def test_kernel_vector_identical_molecules(same_configuration_set):
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    kernel_vector = descriptor.kernel_vector(
        same_configuration_set[0], same_configuration_set, zeta=4
    )
    assert np.allclose(kernel_vector, np.ones_like(kernel_vector), atol=1e-5)


def test_kernel_vector_different_molecules(h2o_configuration, methane):
    descriptor = SoapDescriptor(
        elements=['H', 'C', 'O'], r_cut=5.0, n_max=6, l_max=6, average='inner'
    )
    configurations = ConfigurationSet(h2o_configuration, methane)
    kernel_vector = descriptor.kernel_vector(
        h2o_configuration, configurations, zeta=4
    )
    expected_value = [1.0, 0.29503]
    assert np.allclose(
        kernel_vector, expected_value, atol=1e-3
    ), f'Expected vector {expected_value}, but got {kernel_vector}'
