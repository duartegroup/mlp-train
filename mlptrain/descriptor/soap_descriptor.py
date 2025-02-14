import numpy as np
import mlptrain
from typing import Union, Optional, Sequence
from dscribe.descriptors import SOAP
from mlptrain.descriptor._base import Descriptor


class SoapDescriptor(Descriptor):
    """
    SOAP Descriptor Representation.

    Initializes a SOAP descriptor for computing the Smooth Overlap of Atomic Positions (SOAP) representation.

    Arguments:
        elements (Optional[Sequence[str]]): Atomic species (e.g., `['H', 'O']`) for which the SOAP descriptor is computed.
            If `None`, elements will be inferred from input configurations.
        r_cut (float): Cutoff radius (Å) for considering atomic neighbors, defining the spatial range for SOAP calculations.
        n_max (int): Number of radial basis functions, affecting the resolution in the radial direction.
        l_max (int): Maximum degree of spherical harmonics, controlling the angular resolution.
        average (Optional[str]): Averaging mode for the SOAP descriptor:
            - `"inner"` (default): Averages SOAP vectors before computing the power spectrum.
            - `"outer"`: Computes the power spectrum for each atom, then averages.
            - `None`: No averaging, returns per-atom descriptors.

    Returns:
        np.ndarray: SOAP descriptor matrix of shape `(m, n)`, where:
            - `m` is the number of input configurations.
            - `n` is the descriptor dimensionality, dependent on `n_max` and `l_max`.


    """

    def __init__(
        self,
        elements: Optional[Sequence[str]] = None,
        r_cut: float = 5.0,
        n_max: int = 6,
        l_max: int = 6,
        average: Optional[str] = 'inner',
    ):
        """Initialize SOAP descriptor with given parameters."""
        super().__init__(name='soap_descriptor')
        self.elements = elements
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.average = average

        # Initialize SOAP descriptor if elements are provided
        if self.elements:
            self.soap = SOAP(
                species=self.elements,
                r_cut=self.r_cut,
                n_max=self.n_max,
                l_max=self.l_max,
                average=self.average,
            )
        else:
            self.soap = None

    def compute_representation(
        self,
        configurations: Union[
            mlptrain.Configuration, mlptrain.ConfigurationSet
        ],
    ) -> np.ndarray:
        """Compute the SOAP descriptor matrix for given configurations.

        soap_matrix(config)           -> [[v0, v1, ..]]

        soap_matrix(config1, config2) -> [[v0, v1, ..],
                                   [u0, u1, ..]]

        soap_matrix(configset)        -> [[v0, v1, ..],..]"""

        # Convert single configuration into a list
        if isinstance(configurations, mlptrain.Configuration):
            configurations = [configurations]
        elif isinstance(configurations, mlptrain.ConfigurationSet):
            configurations = configurations.configurations

        # Dynamically set elements if not provided
        if self.soap is None:
            if not self.elements:
                self.elements = list(
                    set(atom.label for c in configurations for atom in c.atoms)
                )

            self.soap = SOAP(
                species=self.elements,
                r_cut=self.r_cut,
                n_max=self.n_max,
                l_max=self.l_max,
                average=self.average,
            )

        soap_vec = self.soap.create(
            [conf.ase_atoms for conf in configurations]
        )
        return soap_vec if soap_vec.ndim > 1 else soap_vec.reshape(1, -1)

    def kernel_vector(
        self,
        configuration: mlptrain.Configuration,
        configurations: Union[
            mlptrain.Configuration, mlptrain.ConfigurationSet
        ],
        zeta=4,
    ) -> np.ndarray:
        """Compute SOAP kernel similarity vector."""

        # Ensure configurations are a list
        if isinstance(configurations, mlptrain.Configuration):
            configurations = [configurations]
        elif isinstance(configurations, mlptrain.ConfigurationSet):
            configurations = configurations.configurations

        v1 = self.compute_representation(configuration)[0]
        m1 = self.compute_representation(configurations)

        # Normalize vectors
        v1 /= np.linalg.norm(v1)
        m1 /= np.linalg.norm(m1, axis=1, keepdims=True)

        return np.power(np.dot(m1, v1), zeta)
