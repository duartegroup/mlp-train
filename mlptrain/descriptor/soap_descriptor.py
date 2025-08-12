import numpy as np
import mlptrain
from typing import Union, Optional, Sequence
from dscribe.descriptors import SOAP
from mlptrain.descriptor._base import Descriptor


class SoapDescriptor(Descriptor):
    """SOAP Descriptor Representation."""

    def __init__(
        self,
        elements: Optional[Sequence[str]] = None,
        r_cut: float = 5.0,
        n_max: int = 6,
        l_max: int = 6,
        average: Optional[str] = 'inner',
    ):
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
                - `None`: No averaging, returns per-atom descriptors."""

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
        """Create a SOAP vector using dscribe (https://github.com/SINGROUP/dscribe)
        for a set of configurations

        soap_vector(config)           -> [[v0, v1, ..]]

        soap_vector(config1, config2) -> [[v0, v1, ..],
                                      [u0, u1, ..]]

        soap_vector(configset)        -> [[v0, v1, ..], ..]

        ---------------------------------------------------------------------------
        Arguments:
        args: Configurations to use


        Returns:
        np.ndarray: SOAP descriptor matrix of shape `(m, n)`, where:
            - `m` is the number of input configurations.
            - `n` is the descriptor dimensionality, dependent on `n_max` and `l_max`.
        """

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

        v1 = self.compute_representation(configuration)
        m1 = self.compute_representation(configurations)

        if self.average in ['inner', 'outer']:
            """ If the averaging mode is 'inner' or 'outer', the SOAP descriptor is averaged over atomic sites to yield a single vector per molecule.
        In this case, the kernel similarity is computed as the dot product of the averaged SOAP vectors, raised to the power of zeta.

        - In **inner averaging**, the expansion coefficients are averaged across atoms *before* computing the power spectrum.
        This is mathematically represented as:

        p ~ sum_m [ (1/n) * sum_i c_nlm(i, Z1) ] * [ (1/n) * sum_i c_n'lm(i, Z2) ]

        -In **outer averaging**, the power spectrum is computed at each atomic site first, and then averaged across all atoms:

        p ~ (1/n) * sum_i sum_m c_nlm(i, Z1) * c_n'lm(i, Z2)"""

            v1 = v1[0]  # Single vector for entire structure
            m1 = m1  # Each row represents one configuration

            # Normalize vectors
            v1 /= np.linalg.norm(v1)
            m1 /= np.linalg.norm(m1, axis=1, keepdims=True)

            return np.power(np.dot(m1, v1), zeta)

        elif self.average == 'off':
            """ Example: Consider a water molecule (H₂O). In a non-averaged SOAP setup, each atom has its own descriptor: one for the oxygen and one for each hydrogen. 
        To compare two water molecules A and B, compute a kernel similarity for each matching pair of atoms:
        K = [k(d_O^A, d_O^B),
             k(d_H1^A, d_H1^B),
             k(d_H2^A, d_H2^B)]
        The overall molecular similarity is then the average of these atomic similarities:
         k_mol(A, B) = (1/3) * [k_O + k_H1 + k_H2]

        More generally, for a molecule with N atoms:
        k_mol(A, B) = (1/N) * sum_i k(d_i^A, d_i^B)
         """

            v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
            m1 /= np.linalg.norm(m1, axis=2, keepdims=True)

            per_atom_similarities = np.einsum(
                'ad,cad->ca', v1, m1
            )  # Compute per-atom kernel similarities
            structure_similarity = np.mean(
                per_atom_similarities, axis=1
            )  # Average per-atom similarities
            structure_similarity = np.power(structure_similarity, zeta)

            return structure_similarity
