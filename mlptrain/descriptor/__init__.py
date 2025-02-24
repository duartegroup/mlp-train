from mlptrain.descriptor.soap_descriptor import SoapDescriptor

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


__all__ = ['SoapDescriptor']
