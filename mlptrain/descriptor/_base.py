from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from mlptrain.log import logger
import mlptrain


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
        configurations: Union[
            mlptrain.Configuration, mlptrain.ConfigurationSet
        ],
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
