import numpy as np
import mlptrain
from autode.atoms import Atom
from typing import Union, Optional, Sequence
from mlptrain.descriptor._base import Descriptor
from ase import io
from ase import Atoms
from io import StringIO
from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main


class ACEDescriptor(Descriptor):
    """ACE Descriptor Representation."""

    def __init__(
        self,
        elements: Optional[Sequence[str]] = None,
        N: int = 3,
        max_deg: int = 6,
        r0: float = 2.3,
        rin: float = 0.1,
        rcut: float = 5.0,
        pin: int = 2,
    ):
        """
        Initializes an ACE descriptor for computing the Atomic Cluster Expansion (ACE) representation.

        Arguments:
            elements (Optional[Sequence[str]]): Atomic species to be used for the ACE basis.
            N (int): (N+1) is the body correlation number, i.e., N=3 means up to 4-body correlations.
            max_deg (int): Maximum polynomial degree for expansion.
            r0 (float): Reference bond length.
            rin (float): Inner cutoff for radial basis.
            rcut (float): Cutoff radius for interactions.
            pin (int): Power exponent for the radial basis.
        """
        super().__init__(name='ACEDescriptor')
        self.elements = elements
        self.N = N
        self.max_deg = max_deg
        self.r0 = r0
        self.rin = rin
        self.rcut = rcut
        self.pin = pin

        # Initialize Julia and ACE1pack
        self.jl = Julia(compiled_modules=False)
        Main.eval(
            'using ACE1pack, LazyArtifacts, MultivariateStats, JuLIP, Glob'
        )

        # Dynamically set basis if elements are provided
        if self.elements:
            self._initialize_basis()
        else:
            self.basis = None

    def _initialize_basis(self):
        """Initializes the ACE basis with the given elements."""
        species_julia = '[:{}]'.format(', :'.join(self.elements))
        Main.eval(
            f"""
        basis = ace_basis(
            species = {species_julia},
            N = {self.N},
            maxdeg = {self.max_deg},
            r0 = {self.r0},
            rin = {self.rin},
            rcut = {self.rcut},
            pin = {self.pin}
        )
        """
        )
        self.basis = Main.eval('basis')

    def compute_representation(
        self,
        configurations: Union[
            mlptrain.Configuration, mlptrain.ConfigurationSet
        ],
    ) -> np.ndarray:
        """
        Compute the ACE descriptor for a set of configurations.

        Handles Extended XYZ format directly to avoid unnecessary ASE conversions.

        Returns:
            np.ndarray: ACE descriptor matrix.
        """
        if isinstance(configurations, mlptrain.Configuration):
            configurations = [
                configurations
            ]  # Convert single configuration to list
            single_config = True
        elif isinstance(configurations, mlptrain.ConfigurationSet):
            single_config = False
        else:
            raise ValueError(
                f'Unsupported configuration type: {type(configurations)}'
            )

        # Dynamically initialize basis if needed
        if self.basis is None:
            if not self.elements:
                self.elements = list(
                    set(atom.label for c in configurations for atom in c.atoms)
                )
            self._initialize_basis()

        ace_vecs = []
        for conf in configurations:
            if isinstance(conf.atoms, Atoms):
                ase_atoms = conf.atoms  # ASE Atoms object is already correct
            elif isinstance(conf.atoms, list) and isinstance(
                conf.atoms[0], Atom
            ):
                # Convert a list of autode Atoms to ASE Atoms
                symbols = [atom.label for atom in conf.atoms]
                positions = [atom.coord for atom in conf.atoms]
                ase_atoms = Atoms(symbols=symbols, positions=positions)
            else:
                raise TypeError(f'Unexpected atoms format: {type(conf.atoms)}')
            if not np.any(ase_atoms.cell):
                ase_atoms.set_cell(
                    [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]
                )

            extxyz_io = StringIO()
            io.write(extxyz_io, ase_atoms, format='extxyz')
            extxyz_string = extxyz_io.getvalue()
            # compute the ace descriptor,averagd method
            Main.eval(
                f'dataset = JuLIP.read_extxyz(IOBuffer("""{extxyz_string}"""))'
            )

            # Compute ACE descriptor
            descriptor = Main.eval(
                """
            descriptors = []
            for atoms in dataset
                for i in 1:length(atoms)
                    descriptor = site_energy(basis, atoms, i)
                    push!(descriptors, descriptor)
                end                 
            end
             return descriptors
            """
            )
            descriptor_np = np.array(
                descriptor, dtype=np.float64
            )  # Shape (num_atoms, descriptor_dim)

            if descriptor_np.ndim == 1:
                descriptor_np = descriptor_np.reshape(1, -1)
            ace_vecs.append(descriptor_np)

        ace_vecs = np.stack(
            ace_vecs, axis=0
        )  # Shape: (num_configs, num_atoms, descriptor_dim)
        return ace_vecs.squeeze() if single_config else ace_vecs

    def kernel_vector(
        self,
        configuration: mlptrain.Configuration,
        configurations: mlptrain.ConfigurationSet,
        zeta: int = 4,
    ) -> np.ndarray:
        """
        Compute similarity kernel between configurations using ACE descriptors.

        Arguments:
            configuration: Single molecular structure.
            configurations: Set of molecular structures.
            zeta (int): Exponent in the kernel function.

        Returns:
            np.ndarray: Kernel similarity vector.
        """
        v1 = self.compute_representation(configuration)  # Shape: (23, 10743)
        m1 = self.compute_representation(
            configurations
        )  # Shape: (2, 23, 10743)
        # Normalize vectors per atom

        v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
        m1 /= np.linalg.norm(m1, axis=2, keepdims=True)

        per_atom_similarities = np.einsum(
            'ad,cad->ca', v1, m1
        )  # Compute per-atom kernel similarities
        structure_similarity = np.mean(
            per_atom_similarities, axis=1
        )  # Average per-atom similarities
        structure_similarity = np.power(structure_similarity, zeta)

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
        return structure_similarity
