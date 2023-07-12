import numpy as np
import mlptrain as mlt
from typing import Union, Optional, Sequence


def soap_matrix(
    *args: Union[mlt.ConfigurationSet, mlt.Configuration],
    elements: Optional[Sequence] = None,
) -> np.ndarray:
    """
    Create a SOAP matrix using dscribe (https://github.com/SINGROUP/dscribe)
    for a set of configurations

    soap_matrix(config)           -> [[v0, v1, ..]]

    soap_matrix(config1, config2) -> [[v0, v1, ..],
                                      [u0, u1, ..]]

    soap_matrix(configset)        -> [[v0, v1, ..],
                                      ..]

    ---------------------------------------------------------------------------
    Arguments:
        args: Configurations to use

        elements: Atomic symbols for which the SOAP matrix should be made

    Returns:
         (np.ndarray): shape = (m, n) for m total configurations
    """
    # NOTE: import within function to allow mlt import without requirement
    from dscribe.descriptors import SOAP

    configurations = []

    for item in args:
        if isinstance(item, mlt.Configuration):
            configurations.append(item)

        elif isinstance(item, mlt.ConfigurationSet):
            configurations.extend([c for c in item])

        else:
            raise ValueError(f"Could not calculate a SOAP vector for {item}")

    # logger.info(f'Calculating SOAP descriptor for {len(configurations)}'
    #             f' configuration(s)')

    if elements is None:
        elements = list(
            set(atom.label for c in configurations for atom in c.atoms)
        )

    # Compute the average SOAP vector where the expansion coefficients are
    # calculated over averages over each site
    soap_desc = SOAP(
        species=elements,
        rcut=5,  # Distance cutoff (Å)
        nmax=6,  # Maximum order of the radial
        lmax=6,  # Maximum order of the angular
        average="inner",
    )

    soap_vec = soap_desc.create([conf.ase_atoms for conf in configurations])
    # logger.info('SOAP calculation done')

    if soap_vec.ndim == 1:
        # soap_desc.create doesn't return a consistent number of dimensions...
        return soap_vec.reshape(1, -1)

    return soap_vec


def soap_kernel_vector(
    configuration: mlt.Configuration,
    configurations: mlt.ConfigurationSet,
    zeta: int = 4,
):
    """
    Calculate the kernel matrix between a set of configurations where the
    kernel is:

    .. math::

        K(p_a, p_b) = (p_a . p_b / (p_a.p_a x p_b.p_b)^1/2 )^ζ

    ---------------------------------------------------------------------------
    Arguments:
        configuration:

        configurations:

        zeta: Power to raise the kernel matrix to

    Returns:
        (np.ndarray): Vector, shape = len(configurations)
    """

    v1 = soap_matrix(configuration)[0]
    v1 /= np.linalg.norm(v1)

    # Normalised matrix over each soap vector (row)
    m1 = soap_matrix(configurations)
    m1 /= np.linalg.norm(m1, axis=1).reshape(len(configurations), 1)

    return np.power(np.dot(m1, v1), zeta)
