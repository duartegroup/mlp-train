import pytest
from autode.atoms import Atom
from mlptrain.descriptor import SoapDescriptor
from mlptrain import Configuration, ConfigurationSet
import numpy as np


@pytest.fixture
def water(h2o):
    return h2o


@pytest.fixture
def configuration_set(water):
    """Fixture to create a ConfigurationSet containing duplicates of a simple molecule."""
    return ConfigurationSet([water, water])


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


def test_compute_representation(water):
    """Test computation of SOAP representation for water"""
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    representation = descriptor.compute_representation(water)
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


def test_kernel_vector_different_molecules(water):
    methane = Configuration(
        atoms=[
            Atom('C', 0, 0, 0),
            Atom('H', 1, 0, 0),
            Atom('H', -1, 0, 0),
            Atom('H', 0, 1, 0),
            Atom('H', 0, -1, 0),
        ]
    )
    descriptor = SoapDescriptor(
        elements=['H', 'C', 'O'], r_cut=5.0, n_max=6, l_max=6, average='inner'
    )
    kernel_vector = descriptor.kernel_vector(water, methane, zeta=4)
    expected_value = 0.8552933998497922
    assert np.testing.assert_allclose(kernel_vector, expected_value, atol=1e-5)
