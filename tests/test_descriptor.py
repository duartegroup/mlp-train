import functools
import torch
torch.load = functools.partial(torch.load, weights_only=False)
import pytest
from autode.atoms import Atom
from mlptrain.descriptor import ACEDescriptor
from mlptrain.descriptor import SoapDescriptor
from mlptrain import Configuration, ConfigurationSet
import numpy as np
from julia.api import Julia

jl = Julia(
    runtime='/home/aleph2/fd/some5167/julia-1.10.2/bin/julia',
    compiled_modules=False,
)


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
def h2o_configuration():
    """Fixture to create a Configuration instance for water."""
    atoms = [
        Atom('O', 0, 0, 0),
        Atom('H', 0.96, 0, 0),
        Atom('H', -0.48, 0.83, 0),
    ]
    return Configuration(atoms=atoms)


def test_ace_descriptor_initialization():
    """Test initialization of ACEDescriptor with and without elements."""
    descriptor_with = ACEDescriptor(
        elements=['H', 'O'], N=3, max_deg=12, rcut=5.0
    )
    assert descriptor_with.elements == ['H', 'O']

    descriptor_without = ACEDescriptor()
    assert descriptor_without.elements is None


def test_compute_representation(h2o_configuration):
    """Test computation of ACE representation for water"""
    descriptor = ACEDescriptor(elements=['H', 'O'], N=3, max_deg=12, rcut=5.0)
    representation = descriptor.compute_representation(h2o_configuration)
    print(representation)
    assert representation.shape[0] > 0, 'Representation should not be empty'
    assert (
        representation.ndim == 2
    ), f'Expected 2D output, got {representation.ndim}D'


def test_kernel_vector_identical_molecules(h2o_configuration):
    """Test ACE kernel for identical molecules (should return ~1.0)"""
    descriptor = ACEDescriptor(elements=['H', 'O'], N=3, max_deg=12, rcut=5.0)
    kernel_vector = descriptor.kernel_vector(
        h2o_configuration, h2o_configuration, zeta=4
    )
    assert np.allclose(kernel_vector, np.ones_like(kernel_vector), atol=1e-5)


def test_kernel_vector_different_molecules(h2o_configuration, methane):
    """Test ACE kernel for different molecules (should be < 1.0)"""
    descriptor = ACEDescriptor(
        elements=['H', 'C', 'O'], N=3, max_deg=12, rcut=5.0
    )
    configurations = ConfigurationSet(h2o_configuration, methane)
    kernel_vector = descriptor.kernel_vector(
        h2o_configuration, configurations, zeta=4
    )
    print(kernel_vector)

    assert (
        kernel_vector[0] > kernel_vector[1]
    ), 'H2O-H2O similarity should be higher than H2O-CH4'


def test_soap_descriptor_initialization_soap():
    """Test initialization of SoapDescriptor with and without elements."""
    # With elements
    descriptor_with = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    assert descriptor_with.elements == ['H', 'O']
    # Without elements (should handle dynamic element setup)
    descriptor_without = SoapDescriptor()
    assert descriptor_without.elements is None


def test_compute_representation_soap(h2o_configuration):
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


def test_kernel_vector_identical_molecules_soap(h2o_configuration):
    descriptor = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6
    )
    kernel_vector = descriptor.kernel_vector(
        h2o_configuration, h2o_configuration, zeta=4
    )
    assert np.allclose(kernel_vector, np.ones_like(kernel_vector), atol=1e-5)


def test_kernel_vector_different_molecules_soap(h2o_configuration, methane):
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
