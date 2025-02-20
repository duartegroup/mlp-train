import numpy as np
import pytest
from mlptrain.descriptor import (
    SoapDescriptor,
    soap_matrix,
    soap_kernel_vector,
)


def test_imports():
    """Test that soap_matrix and soap_kernel_vector can be imported and are callable."""
    assert callable(soap_matrix)
    assert callable(soap_kernel_vector)


def test_soap_descriptor_initialization():
    """Test the initialization of the SoapDescriptor class."""
    soap_desc = SoapDescriptor(
        elements=['H', 'O'], r_cut=5.0, n_max=6, l_max=6, average='inner'
    )
    assert soap_desc.elements == ['H', 'O']
    assert soap_desc.r_cut == 5.0
    assert soap_desc.n_max == 6
    assert soap_desc.l_max == 6
    assert soap_desc.average == 'inner'


@pytest.fixture
def mock_configuration():
    """Mock configuration for testing."""

    class MockConfiguration:
        # Simulate the necessary properties and methods
        def __init__(self):
            self.ase_atoms = np.array([])

    return MockConfiguration()


def test_compute_representation(mock_configuration):
    """Test compute_representation with a mock configuration."""
    soap_desc = SoapDescriptor(elements=['H', 'O'])
    # Simulating a configuration set with empty ASE atoms
    result = soap_desc.compute_representation([mock_configuration])
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 0)


def test_kernel_vector(mock_configuration):
    """Test kernel_vector computation."""
    soap_desc = SoapDescriptor(elements=['H', 'O'])
    result = soap_desc.kernel_vector(mock_configuration, [mock_configuration])
    assert isinstance(result, np.ndarray)
    # Check the result shape
    assert result.shape == (1,)
