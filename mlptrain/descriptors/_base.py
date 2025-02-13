from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from mlptrain.log import logger
import mlptrain as mlt


class Descriptor(ABC):
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
    def compute_representation(
        self,
        configurations: Union[mlt.Configuration, mlt.ConfigurationSet],
    ) -> np.ndarray:
        """
        Compute descriptor representation for a given molecular configuration.

        Arguments:
            configuration: A molecular structure (e.g., `mlptrain.Configuration`).
        Returns:
            np.ndarray: The computed descriptor representation as a vector/matrix.
        """

    @abstractmethod
    def kernel_vector(
        self, configuration, configurations, zeta: int = 4
    ) -> np.ndarray:
        """Calculate the kernel matrix between a set of configurations where the  kernel is: .. math::

        K(p_a, p_b) = (p_a . p_b / (p_a.p_a x p_b.p.b)^1/2 )^ζ

        ---------------------------------------------------------------------------
        Arguments:
            configuration:

            configurations:

            zeta: Power to raise the kernel matrix to

        Returns:
            (np.ndarray): Vector, shape = len(configurations)"""

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

    def average(self, average_method: str = 'no_average'):
        """
        Compute the average of the descriptor representation to accommodate systems of different sizes.

        Arguments:
            average_method (str): Specifies the averaging method:
                              - "inner" (default), "outer", or "no_average" for soap_descriptor
                              - "average" or "no_average" (default) for ace_descriptor
                              - No parameter needed for mace_descriptor
        """
