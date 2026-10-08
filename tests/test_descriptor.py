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


def test_kernel_vector_identical_molecules(h2o_configuration):
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    kernel_vector = descriptor.kernel_vector(
        h2o_configuration, h2o_configuration, zeta=4
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
    assert np.allclose(kernel_vector, expected_value, atol=1e-3), (
        f'Expected vector {expected_value}, but got {kernel_vector}'
    )


@pytest.fixture
def water():
    atoms = [
        Atom('O', 0, 0, 0),
        Atom('H', 0.96, 0, 0),
        Atom('H', -0.24, 0.93, 0),
    ]
    return Configuration(atoms=atoms)


@pytest.fixture
def distorted_water():
    atoms = [
        Atom('O', 0, 0, 0),
        Atom('H', 0.9, 0.1, 0),
        Atom('H', -0.3, 0.9, 0),
    ]
    return Configuration(atoms=atoms)


def test_compute_representation_no_average_single_config(water):
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6, average='off'
    )
    representation = descriptor.compute_representation(water)
    assert representation.shape == (1, 3, 546)


def test_kernel_vector_no_average_single_config(water, distorted_water):
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6, average='off'
    )
    kernel_vector = descriptor.kernel_vector(
        water, ConfigurationSet(water), zeta=4
    )
    assert np.allclose(kernel_vector, [1.0], atol=1e-5)

    kernel_vector = descriptor.kernel_vector(
        water, ConfigurationSet(distorted_water), zeta=4
    )
    assert kernel_vector.shape == (1,)
    assert 0.0 < kernel_vector[0] < 1.0


def test_kernel_vector_no_average_multiple_configs(water, distorted_water):
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6, average='off'
    )
    configurations = ConfigurationSet(water, distorted_water)
    kernel_vector = descriptor.kernel_vector(water, configurations, zeta=4)
    assert kernel_vector.shape == (2,)
    assert np.isclose(kernel_vector[0], 1.0, atol=1e-5)
    assert kernel_vector[1] < 1.0


def test_kernel_vector_no_average_different_atoms(water, methane):
    descriptor = SoapDescriptor(
        elements=['H', 'C', 'O'], r_cut=5.0, n_max=6, l_max=6, average='off'
    )
    with pytest.raises(ValueError):
        descriptor.kernel_vector(water, ConfigurationSet(water, methane))

    with pytest.raises(ValueError):
        descriptor.kernel_vector(water, ConfigurationSet(methane))


def test_kernel_vector_no_average_different_atom_order(water):
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6, average='off'
    )
    reordered = Configuration(
        atoms=[water.atoms[1], water.atoms[0], water.atoms[2]]
    )
    with pytest.raises(ValueError):
        descriptor.kernel_vector(water, ConfigurationSet(reordered))
