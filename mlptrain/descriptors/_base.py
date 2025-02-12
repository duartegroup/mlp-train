from abc import ABC, abstractmethod
import numpy as np
import logging
import mlptrain as mlp

# Setup logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DescriptorBase(ABC):
    """Abstract base class for molecular feature descriptors."""

    def __init__(self, name: str):
        """
        Initializes the descriptor representation.

        Arguments:
            name (str): Name of the descriptor. e.g., "ace_descriptor","soap_descriptor","mace_descriptor"
        """
        self.name = str(name)
        logger.info(f'Initialized {self.name} descriptor.')

    @abstractmethod
    def compute(self, configurations: mlp.ConfigurationSet) -> np.ndarray:
        """
        Compute descriptor representation for a given molecular configuration.

        Arguments:
            configuration: A molecular structure (e.g., `mlptrain.Configuration`).
        Returns:
            np.ndarray: The computed descriptor representation as a vector/matrix.
        """
        pass


@abstractmethod
def kernel_vector(
    self, configuration, configurations, zeta: int = 4
) -> np.ndarray:
    """
    Calculate the kernel matrix between a set of configurations where the
    kernel is:

    .. math::

        K(p_a, p_b) = (p_a . p_b / (p_a.p_a x p_b.p.b)^1/2 )^ζ

    ---------------------------------------------------------------------------
    Arguments:
        configuration:

        configurations:

        zeta: Power to raise the kernel matrix to

    Returns:
        (np.ndarray): Vector, shape = len(configurations)
    """
    pass


def normalize(self, vector: np.ndarray) -> np.ndarray:
    """
    Normalize a feature vector to unit norm.

    Arguments:
        vector (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector.
    """
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm
