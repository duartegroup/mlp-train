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
        super().__init__(name='SoapDescriptor')
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
            self.soap = None  # To be initialized dynamically

    def compute_representation(
        self,
        configurations: Union[
            mlptrain.Configuration, mlptrain.ConfigurationSet
        ],
    ) -> np.ndarray:
        """C   Create a SOAP vector using dscribe (https://github.com/SINGROUP/dscribe)
        for a set of configurations

        soap_vector(config)           -> [[v0, v1, ..]]

        soap_vector(config1, config2) -> [[v0, v1, ..],
                                      [u0, u1, ..]]

        soap_vector(configset)        -> [[v0, v1, ..], ..]

        ---------------------------------------------------------------------------
         Arguments:
        args: Configurations to use


        Returns:
         (np.ndarray): shape = (m, n) for m total configurations"""

        if isinstance(configurations, mlptrain.Configuration):
            configurations = [
                configurations
            ]  # Convert to list if it's a single Configuration
        elif not isinstance(configurations, mlptrain.ConfigurationSet):
            raise ValueError(
                f'Unsupported configuration type: {type(configurations)}'
            )

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
        configurations: mlptrain.ConfigurationSet,
        zeta: int = 4,
    ) -> np.ndarray:
        """Calculate the kernel matrix between a set of configurations where the
        kernel is:

        .. math::

            K(p_a, p_b) = (p_a . p_b / (p_a.p_a x p_b.p_b)^1/2 )^ζ

        ---------------------------------------------------------------------------
        Arguments:
            configuration:

            configurations:

            zeta: Power to raise the kernel matrix to

        Returns:
            (np.ndarray): Vector, shape = len(configurations)"""

        v1 = self.compute_representation(configuration)[0]
        m1 = self.compute_representation(configurations)

        # Normalize vectors using the defined normalize function from base.py
        v1 = Descriptor.normalize(v1)
        m1 = np.array([Descriptor.normalize(vec) for vec in m1])

        # Compute the kernel using the normalized vectors
        kernel_values = np.power(np.dot(m1, v1), zeta)
        return kernel_values
