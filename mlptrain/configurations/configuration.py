import mlptrain
import ase
import numpy as np
from typing import Optional, Union, List
from copy import deepcopy
from autode.atoms import AtomCollection, Atom
import autode.atoms
import ase.atoms
from mlptrain.log import logger
from mlptrain.energy import Energy
from mlptrain.forces import Forces
from mlptrain.box import Box
from mlptrain.configurations.calculate import run_autode
from scipy.spatial import cKDTree
import random
import autode as ade
from math import dist
from autode.solvent.solvents import get_solvent


class Configuration(AtomCollection):
    """Configuration of atoms"""

    def __init__(
        self,
        atoms: Union[autode.atoms.Atoms, List[Atom], None] = None,
        charge: int = 0,
        mult: int = 1,
        box: Optional[Box] = None,
    ):
        """
        Set of atoms perhaps in a periodic box with an overall charge and
        spin multiplicity.

        May contain a list with the indices of each first atom of a molecule, if there are several.
        When adding several different molecules to a configuration, one can add the
        starting index of each molecule to this list. e.g. a list of three waters
        would be [0, 3, 6]. This list gets updated when the solvate function is called.

        -----------------------------------------------------------------------
        Arguments:
            atoms:
            charge:
            mult:
            box: Optional box, if None
            mol_list: List[int] = None

        """
        super().__init__(atoms=atoms)

        self.charge = charge
        self.mult = mult
        self.box = box

        self.energy = Energy()
        self.forces = Forces()

        # Collective variable values (obtained using PLUMED)
        self.plumed_coordinates: Optional[np.ndarray] = None

        self.time: Optional[float] = None  # Time in a trajectory
        self.n_ref_evals = 0  # Number of reference evaluations

    @property
    def ase_atoms(self) -> 'ase.atoms.Atoms':
        """
        ASE atoms for this configuration, absent of energy  and force
        properties.

        -----------------------------------------------------------------------
        Returns:
            (ase.atoms.Atoms): ASE atoms
        """
        _atoms = ase.atoms.Atoms(
            symbols=[atom.label for atom in self.atoms],
            positions=self.coordinates,
            pbc=self.box is not None,
        )

        if self.box is not None:
            _atoms.set_cell(cell=self.box.size)

        return _atoms

    def solvate(
        self,
        box_size: float = None,
        buffer_distance: float = 10,
        solvent_name: str = None,
        solvent_density: float = None,
        solvent_molecule: ade.Molecule = None,
        threshold: float = 1.8,
    ) -> None:
        """Solvate the configuration of a solute using solvent molecules.
        The box size can be specified manually in Å or it can be calculated automatically
        by adding a buffer distance to the maximum distance between any two atoms in the solute.
        The solvent can be specified either by name, if it is already contained
        in the solvent database or by providing an autode Molecule object, in which
        case the density of the solvent must also be provided.

        ___________________________________________________________________________

        Arguments:

        box_size: float = None
            The size of the box in Å. If None, the box size will be calculated automatically.

        buffer_distance: float = 10
            The distance in Å to be added to the maximum distance between any two atoms in the solute.

        solvent_name: str = None
            The name of the solvent contained in the solvent database.

        solvent_molecule: autode.Molecule = None
            The autode Molecule object representing the solvent, if provided explicitly by the user.

        solvent_density: float = None
            The density which must be provided along with the solvent molecule.

        threshold: float = 1.8
            The distance in Å below which two atoms are considered to be in contact.

        ___________________________________________________________________________

        """
        # Calculate the box size if not provided, based on the maximum distance between any two atoms in the solute
        # and the buffer distance
        if box_size is None:
            box_size = get_max_mol_distance(self.atoms) + buffer_distance

        # Assume cubic box
        self.box = Box([box_size] * 3)

        # If both the solvent name and the solvent molecule and density are provided, stop checking
        if None not in (solvent_density, solvent_molecule):
            pass

        # If the solvent name is provided, get the solvent molecule and density from the solvent database
        # by getting the smiles from autode's solvent database, creating a molecule object and optimising it
        # with xtb, then get the density from the density database
        elif solvent_name is not None:
            solvent = get_solvent(solvent_name, kind='implicit')
            solvent_smiles = solvent.smiles
            solvent_molecule = ade.Molecule(smiles=solvent_smiles)
            solvent_molecule.optimise(method=ade.methods.XTB())
            solvent_density = solvent_densities[solvent.name]

        else:
            # If neither the solvent name nor the solvent molecule and density are provided, raise an error
            raise ValueError(
                'Either the solvent name or the solvent molecule and density must be provided'
            )

        # Move solute to the box middle
        solute_com = self.com
        for n, atom in enumerate(self.atoms):
            atom.coordinate = atom.coordinate - solute_com + (box_size / 2)

        # Move the solvent to the box origin, so that the random vectors added later are all within the box
        solvent_com = solvent_molecule.com
        for n, atom in enumerate(solvent_molecule.atoms):
            atom.coordinate = atom.coordinate - solvent_com

        # Calculate the number of solvent molecules to be inserted
        solvent_mass = sum([atom.mass for atom in solvent_molecule.atoms])
        # calculate the theoretical volume a single molecule should take up at its experimentally determined density
        # by finding the mass of a single molecule in g, dividing by the density in g/cm^3 and then converting
        # to nm by multiplying by 1e24 to get Å
        single_sol_volume = (solvent_mass / 6.02214e23) / (
            solvent_density / 1e24
        )

        solvent_number = int(np.round((box_size**3) / single_sol_volume, 0))

        self.k_d_tree_insertion(
            solvent_molecule, box_size, threshold, solvent_number
        )

    def k_d_tree_insertion(
        self,
        solvent_molecule: ade.Molecule,
        box_size: float,
        threshold: float,
        n_solvent: int,
    ) -> np.ndarray:
        """Insert solvent molecules into the box using a k-d tree to check for collisions.
        Implemented according to the algorithm described in the paper:

        "https://chemrxiv.org/engage/chemrxiv/article-details/678621ccfa469535b9ea8786"

        ___________________________________________________________________________

        Arguments:

        solvent_molecule: autode.Molecule
            The molecule representing the solvent to be inserted.

        box_size: float
            The size of the box in Å. A cubic box is assumed, and boxes where the three box
            vectors are not equal are not supported.

        threshold: float
            The distance in Å below which two atoms are considered to be in contact.

        n_solvent: int
            The number of solvent molecules to be inserted.

        ___________________________________________________________________________
        """

        # Get the coordinates of the solute in the middle of the box
        system_coords = np.array([atom.coordinate for atom in self.atoms])
        # Get the coordinates of the single isolated solvent molecule
        solvent_coords = np.array(
            [atom.coordinate for atom in solvent_molecule.atoms]
        )
        # Initialise a list to keep track of the inserted solvent molecules
        mol_list = [0, len(solvent_coords)]

        for i in range(n_solvent):
            # Build a k-d tree from the system coordinates in order to query the nearest neighbours later
            existing_tree = build_cKDTree(system_coords)
            inserted = False
            attempt = 0

            # Try to insert the solvent molecule into the box with a maximum of 1000 attempts
            while not inserted and attempt < 1000:
                attempt += 1

                # Generate a random rotation matrix and a random translation vector
                rot_matrix = random_rotation()
                rot_solvent = np.dot(solvent_coords, rot_matrix)
                translation = random_vector_in_box(box_size)

                # Translate the rotated solvent molecule and check if it is within the box
                trial_coords = rot_solvent + translation
                if not np.all(
                    (
                        [
                            np.all(coord < box_size) and np.all(coord > 0)
                            for coord in trial_coords
                        ]
                    )
                ):
                    continue

                # Query the nearest neighbours of the trial coordinates and check if they are within the threshold
                # If they are, add them to the system coordinates and update the mol_list
                distances, indeces = existing_tree.query(trial_coords)
                if all(distances > threshold):
                    solvent_translated = deepcopy(solvent_molecule)
                    for n, atom in enumerate(solvent_translated.atoms):
                        atom.coordinate = trial_coords[n]
                    self.atoms.extend(solvent_translated.atoms)
                    system_coords = np.concatenate(
                        (system_coords, trial_coords)
                    )
                    inserted = True
                    mol_list.append(len(self.atoms))

        self.mol_list = mol_list
        return system_coords

    def update_attr_from(self, configuration: 'Configuration') -> None:
        """
        Update system attributes from a configuration

        -----------------------------------------------------------------------
        Arguments:
            configuration:
        """

        self.charge = configuration.charge
        self.mult = configuration.mult
        self.box = deepcopy(configuration.box)

        return None

    def save_xyz(
        self,
        filename: str,
        append: bool = False,
        true: bool = False,
        predicted: bool = False,
    ) -> None:
        """
        Print this configuration as an extended xyz file where the first 4
        columns are the atom symbol, x, y, z and, if this configuration
        contains forces then add the x, y, z components of the force on as
        columns 4-7.

        -----------------------------------------------------------------------
        Arguments:
            filename:

            append: (bool) Append to the end of this xyz file?

            true: Save the true energy and forces

            predicted: Save the predicted energy and forces
        """
        # logger.info(f'Saving configuration to {filename}')

        a, b, c = [0.0, 0.0, 0.0] if self.box is None else self.box.size

        if true and predicted:
            raise ValueError(
                'Cannot save both predicted and true '
                f'quantities to {filename}'
            )

        if not (true or predicted):
            prop_str = ''

        else:
            energy = self.energy.predicted if predicted else self.energy.true
            prop_str = f'energy={energy if energy is not None else 0.:.8f} '

            prop_str += 'Properties=species:S:1:pos:R:3'
            forces = self.forces.predicted if predicted else self.forces.true
            if forces is not None:
                prop_str += ':forces:R:3'

        if not filename.endswith('.xyz'):
            logger.warning('Filename had no .xyz extension - adding')
            filename += '.xyz'

        with open(filename, 'a' if append else 'w') as exyz_file:
            print(
                f'{len(self.atoms)}\n'
                f'Lattice="{a:.6f} 0.000000 0.000000 '
                f'0.000000 {b:.6f} 0.000000 '
                f'0.000000 0.000000 {c:.6f}" '
                f'{prop_str}',
                file=exyz_file,
            )

            for i, atom in enumerate(self.atoms):
                x, y, z = atom.coord
                line = f'{atom.label} {x:.5f} {y:.5f} {z:.5f} '

                if (true or predicted) and forces is not None:
                    fx, fy, fz = forces[i]
                    line += f'{fx:.5f} {fy:.5f} {fz:.5f}'

                print(line, file=exyz_file)

        return None

    def single_point(
        self,
        method: Union[str, 'mlptrain.potentials._base.MLPotential'],
        n_cores: int = 1,
    ) -> None:
        """
        Run a single point energy and gradient (force) evaluation using
        either a reference method defined by a string (e.g. 'orca') or a
        machine learned potential (with a .predict) method.

        -----------------------------------------------------------------------
        Arguments:
            method:

            n_cores: Number of cores to use for the calculation
        """
        implemented_methods = ['xtb', 'orca', 'g09', 'g16']

        if isinstance(method, str) and method.lower() in implemented_methods:
            run_autode(self, method, n_cores=n_cores)
            self.n_ref_evals += 1
            return None

        elif hasattr(method, 'predict'):
            method.predict(self)

        else:
            raise ValueError(
                f'Cannot use {method} to predict energies and ' f'forces'
            )

        return None

    def __eq__(self, other) -> bool:
        """Another configuration is identical to this one"""
        eq = (
            isinstance(other, Configuration)
            and other.n_atoms == self.n_atoms
            and other.mult == self.mult
            and other.charge == self.charge
            and other.box == self.box
        )

        if eq and self.n_atoms > 0:
            rmsd = np.linalg.norm(self.coordinates - other.coordinates)
            return eq and rmsd < 1e-10
        return eq

    def copy(self) -> 'Configuration':
        return deepcopy(self)


def random_rotation() -> np.ndarray:
    """Generate a random rotation matrix"""

    theta = random.random() * 360
    kappa = random.random() * 360
    gamma = random.random() * 360

    rot_matrix = np.eye(3)
    rot_matrix = np.dot(
        rot_matrix,
        np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        ),
    )
    rot_matrix = np.dot(
        rot_matrix,
        np.array(
            [
                [np.cos(kappa), 0, np.sin(kappa)],
                [0, 1, 0],
                [-np.sin(kappa), 0, np.cos(kappa)],
            ]
        ),
    )
    rot_matrix = np.dot(
        rot_matrix,
        np.array(
            [
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1],
            ]
        ),
    )

    return rot_matrix


def random_vector_in_box(box_size: float) -> np.ndarray:
    """Generate a random vector in a box"""

    return np.array([random.random() * box_size for i in range(3)])


def build_cKDTree(coords: np.ndarray) -> cKDTree:
    """Build a cKDTree from a set of coordinates"""

    return cKDTree(coords)


def get_max_mol_distance(conf_atoms: List[Atom]) -> float:
    return max(
        [
            dist(atom1.coordinate, atom2.coordinate)
            for atom1 in conf_atoms
            for atom2 in conf_atoms
        ]
    )


solvent_densities = {
    'water': 1.0,
    'dichloromethane': 1.33,
    'acetone': 0.79,
    'acetonitrile': 0.786,
    'benzene': 0.874,
    'trichloromethane': 1.49,
    'cs2': 1.26,
    'dmf': 0.944,
    'dmso': 1.10,
    'diethyl ether': 0.713,
    'methanol': 0.791,
    'hexane': 0.655,
    'thf': 0.889,
    'toluene': 0.867,
    'acetic acid': 1.05,
    '1-butanol': 0.81,
    '2-butanol': 0.808,
    'ethanol': 0.789,
    'heptane': 0.684,
    'pentane': 0.626,
    '1-propanol': 0.803,
    'pyridine': 0.982,
    'ethyl acetate': 0.902,
    'cyclohexane': 0.779,
    'carbon tetrachloride': 1.59,
    'chlorobenzene': 1.11,
    '1,2-dichlorobenzene': 1.30,
    'n,n-dimethylacetamide': 0.937,
    'dioxane': 1.03,
    '1,2-ethanediol': 1.11,
    'decane': 0.73,
    'dibromomethane': 2.50,
    'dibutylether': 0.764,
    '1-bromopropane': 1.35,
    '2-bromopropane': 1.31,
    '1-chlorohexane': 0.88,
    '1-chloropentane': 0.88,
    '1-chloropropane': 0.89,
    'diethylamine': 0.707,
    '1-decanol': 0.83,
    'diiodomethane': 3.33,
    '1-fluorooctane': 0.88,
    '1-heptanol': 0.82,
    '1-hexanol': 0.814,
    '1-hexene': 0.673,
    '1-hexyne': 0.715,
    '1-iodobutane': 1.62,
    '1-iodohexadecane': 1.26,
    '1-iodopentane': 1.52,
    '1-iodopropane': 1.75,
    'dipropylamine': 0.738,
    'n-dodecane': 0.75,
    '1-nitropropane': 1.00,
    'ethanethiol': 0.839,
    '1-nonanol': 0.83,
    '1-octanol': 0.83,
    '1-pentanol': 0.814,
    '1-pentene': 0.64,
    'ethyl benzene': 0.867,
    '2,2,2-trifluoroethanol': 1.39,
    'fluorobenzene': 1.02,
    '2,2,4-trimethylpentane': 0.69,
    'formamide': 1.13,
    '2,4-dimethylpentane': 0.67,
    '2,4-dimethylpyridine': 0.93,
    '2,6-dimethylpyridine': 0.93,
    'n-hexadecane': 0.77,
    'dimethyl disulfide': 1.06,
    'ethyl methanoate': 0.92,
    'ethyl phenyl ether': 0.97,
    'formic acid': 1.22,
    'hexanoic acid': 0.93,
    '2-chlorobutane': 0.87,
    '2-heptanone': 0.81,
    '2-hexanone': 0.81,
    '2-methoxyethanol': 0.96,
    '2-methyl-1-propanol': 0.80,
    '2-methyl-2-propanol': 0.79,
    '2-methylpentane': 0.65,
    '2-methylpyridine': 0.95,
    '2-nitropropane': 1.00,
    '2-octanone': 0.82,
    '2-pentanone': 0.81,
    'iodobenzene': 1.83,
    'iodoethane': 1.93,
    'iodomethane': 2.28,
    'isopropylbenzene': 0.86,
    'p-isopropyltoluene': 0.86,
    'mesitylene': 0.86,
    'methyl benzoate': 1.09,
    'methyl butanoate': 0.90,
    'methyl ethanoate': 0.93,
    'methyl methanoate': 0.97,
    'methyl propanoate': 0.91,
    'n-methylaniline': 0.99,
    'methylcyclohexane': 0.77,
    'n-methylformamide (e/z mixture)': 1.01,
    'nitrobenzene': 1.20,
    'nitroethane': 1.05,
    'nitromethane': 1.14,
    'o-nitrotoluene': 1.16,
    'n-nonane': 0.72,
    'n-octane': 0.70,
    'n-pentadecane': 0.77,
    'pentanal': 0.81,
    'pentanoic acid': 0.94,
    'pentyl ethanoate': 0.88,
    'pentyl amine': 0.74,
    'perfluorobenzene': 1.61,
    'propanal': 0.81,
    'propanoic acid': 0.99,
    'propanenitrile': 0.78,
    'propyl ethanoate': 0.89,
    'propyl amine': 0.72,
    'tetrachloroethene': 1.62,
    'tetrahydrothiophene-s,s-dioxide': 1.26,
    'tetralin': 0.97,
    'thiophene': 1.06,
    'thiophenol': 1.07,
    'tributylphosphate': 0.98,
    'trichloroethene': 1.46,
    'triethylamine': 0.73,
    'n-undecane': 0.74,
    'xylene mixture': 0.86,
    'm-xylene': 0.86,
    'o-xylene': 0.88,
    'p-xylene': 0.86,
    '2-propanol': 0.785,
    '2-propen-1-ol': 0.85,
    'e-2-pentene': 0.65,
    '3-methylpyridine': 0.96,
    '3-pentanone': 0.81,
    '4-heptanone': 0.81,
    '4-methyl-2-pentanone': 0.80,
    '4-methylpyridine': 0.96,
    '5-nonanone': 0.82,
    'benzyl alcohol': 1.04,
    'butanoic acid': 0.96,
    'butanenitrile': 0.80,
    'butyl ethanoate': 0.88,
    'butylamine': 0.74,
    'n-butylbenzene': 0.86,
    'sec-butylbenzene': 0.86,
    'tert-butylbenzene': 0.86,
    'o-chlorotoluene': 1.08,
    'm-cresol': 1.03,
    'o-cresol': 1.07,
    'cyclohexanone': 0.95,
    'isoquinoline': 1.10,
    'quinoline': 1.09,
    'argon': 0.001784,
    'krypton': 0.003733,
    'xenon': 0.005894,
}
