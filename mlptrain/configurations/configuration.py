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
        contact_threshold: float = 1.8,
    ) -> None:
        """Solvate the configuration of a solute using solvent molecules. Currently solvent mixtures are not supported.
        The box size can be specified manually in Å or it can be calculated automatically
        by adding a buffer distance (in Å) to the maximum distance between any two atoms in the solute.

        The solvent can be specified either by name, if it is already contained
        in the solvent database. With this option, the solvent will be optimised
        by XTB and the density will be extracted from the database.

        Another option is to providing an autode Molecule object of a pre-optimised solvent molecule
        of your choosing, in which case the density of the solvent (in g/cm^3)
        must also be provided. This can also serve to provide non-standard solvents
        and densities.

        The contact threshold is the distance in Å below which two atoms
        are considered to be in contact, when placing the solvents in the box.

        ___________________________________________________________________________

        Arguments:

        box_size: float = None
            The size of the box in Å. If None, the box size will be calculated automatically. This is done
            by calculating the maximum distance between any two atoms in the solute and adding a buffer distance,
            which is specified by the buffer_distance argument.

        buffer_distance: float = 10
            The distance in Å to be added to the maximum distance between any two atoms in the solute, to calculate
            the box size. This argument is only used if the box_size is not provided.

        solvent_name: str = None
            The name of the solvent contained in the solvent database.

        solvent_molecule: autode.Molecule = None
            The autode Molecule object representing the solvent, if provided explicitly by the user.

        solvent_density: float = None
            The density, in g/cm^3 which must be provided along with the solvent molecule.

        contact_threshold: float = 1.8
            The distance in Å below which two atoms are considered to be in contact.

        ___________________________________________________________________________

        """
        # Calculate the box size if not provided, based on the maximum distance between any two atoms in the solute
        # and the buffer distance
        if box_size is None:
            box_size = _get_max_mol_distance(self.atoms) + buffer_distance

        # Assume cubic box
        self.box = Box([box_size] * 3)

        # If both solvent molecule and density are provided, stop checking
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
            # If neither the solvent name nor the combination of solvent molecule and density are provided, raise an error
            raise ValueError(
                'Either the solvent name or the combination of solvent molecule and density must be provided'
            )

        # Move solute to the box center
        solute_com = self.com
        for n, atom in enumerate(self.atoms):
            atom.coordinate = atom.coordinate - solute_com + (box_size / 2)

        # Move the solvent to the box origin, so that the random vectors added later are all within the box
        solvent_com = solvent_molecule.com
        for n, atom in enumerate(solvent_molecule.atoms):
            atom.coordinate = atom.coordinate - solvent_com

        # Calculate the number of solvent molecules to be inserted
        solvent_mass = sum([atom.mass for atom in solvent_molecule.atoms])
        # Calculate the volume of a single solvent molecule by first calculating the mass of a single molecule: m = M/N_a
        single_mol_mass = solvent_mass / 6.02214e23
        # 1e24 is used to convert the density from g/cm^3 to g/Å^3 (1e24 Å^3 = 1 cm^3, 1e8 Å = 1 cm)
        density_in_angstrom = solvent_density / 1e24
        # Then calculate the volume in Å^3 of a single molecule by dividing the mass by the density
        single_sol_volume = (single_mol_mass) / (density_in_angstrom)
        # Number of solvent molecules that would fit into the box without the solute
        solvent_number = int(np.round((box_size**3) / single_sol_volume, 0))

        logger.info(
            f'Attempting to add {solvent_number} solvent molecules'
            f'with the formula {solvent_molecule.formula} to a cubic'
            f'box with a side length of {box_size:.2f} Å'
        )

        self.k_d_tree_insertion(
            solvent_molecule, box_size, contact_threshold, solvent_number
        )

    def k_d_tree_insertion(
        self,
        solvent_molecule: ade.Molecule,
        box_size: float,
        contact_threshold: float,
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

        contact_threshold: float
            The distance in Å below which two atoms are considered to be in contact.

        n_solvent: int
            The number of solvent molecules to be inserted.

        ___________________________________________________________________________
        """

        # Get the coordinates of the solute in the center of the box
        system_coords = np.array([atom.coordinate for atom in self.atoms])
        # Get the coordinates of the single isolated solvent molecule
        solvent_coords = np.array(
            [atom.coordinate for atom in solvent_molecule.atoms]
        )
        # Initialise a list to keep track of the inserted solvent molecules, with each element
        # being the index of the first atom of a molecule
        mol_list = [0, len(system_coords)]
        # Initialise a count of the solvents inserted into the box. The ideal solvent number is
        # the number of solvent molecules that fit into the box, but this number might not be reached
        # because the solute might be in the way of some of the solvent molecules
        solvents_inserted = 0

        for i in range(n_solvent):
            # Build a k-d tree from the system coordinates in order to query the nearest neighbours later
            existing_tree = _build_cKDTree(system_coords)
            inserted = False
            attempt = 0

            # Try to insert the solvent molecule into the box with a maximum of 1000 attempts
            while not inserted and attempt < 1000:
                attempt += 1

                # Generate a random rotation matrix and a random translation vector which
                # depends on a seed number that is the product of the current iteration (number of solvent
                # being added) to the third power and the attempt number. This is to avoid
                # the same solvent molecule being inserted multiple times in the same orientation
                seed_number = i**3 * attempt
                rot_matrix = _random_rotation(seed_number)
                rot_solvent = np.dot(solvent_coords, rot_matrix)
                translation = _random_vector_in_box(box_size, seed_number)

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

                # Query the nearest neighbours of the trial coordinates and check if they are within the contact_threshold
                # If they are, add them to the system coordinates and update the mol_list
                distances, indeces = existing_tree.query(trial_coords)
                if all(distances > contact_threshold):
                    solvent_translated = deepcopy(solvent_molecule)

                    for n, atom in enumerate(solvent_translated.atoms):
                        atom.coordinate = trial_coords[n]

                    self.atoms.extend(solvent_translated.atoms)
                    system_coords = np.concatenate(
                        (system_coords, trial_coords)
                    )
                    inserted = True
                    mol_list.append(len(self.atoms))
                    solvents_inserted += 1

        logger.info(
            f'Inserted {solvents_inserted} solvent molecules into the box'
        )

        # remove the last element of the mol_list, as this is the index of the last atom in the system
        self.mol_list = mol_list[:-1]
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


def _random_rotation(seed: int) -> np.ndarray:
    """Generate a random rotation matrix"""
    random.seed(seed)
    theta = random.random() * 360
    random.seed(seed * 2)
    kappa = random.random() * 360
    random.seed(seed * 3)
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


def _random_vector_in_box(box_size: float, seed: int) -> np.ndarray:
    """Generate a random vector in a box"""
    random.seed(seed)
    return np.array([random.random() * box_size for i in range(3)])


def _build_cKDTree(coords: np.ndarray) -> cKDTree:
    """Build a cKDTree from a set of coordinates"""

    return cKDTree(coords)


def _get_max_mol_distance(conf_atoms: List[Atom]) -> float:
    return max(
        [
            dist(atom1.coordinate, atom2.coordinate)
            for atom1 in conf_atoms
            for atom2 in conf_atoms
        ]
    )


# Solvent densities were taken from the CRC handbook
# CRC Handbook of Chemistry and Physics, 97th ed.; Haynes, W. M., Ed.;
# CRC Press: Boca Raton, FL, 2016; ISBN 978-1-4987-5429-3.
solvent_densities = {
    'water': 1.0,  # CRC handbook
    'dichloromethane': 1.33,  # CRC handbook
    'acetone': 0.79,  # CRC handbook
    'acetonitrile': 0.786,  # CRC handbook
    'benzene': 0.874,  # CRC handbook
    'trichloromethane': 1.49,  # CRC handbook
    'cs2': 1.26,  # CRC handbook
    'dmf': 0.944,  # CRC handbook
    'dmso': 1.10,  # CRC handbook
    'diethyl ether': 0.713,  # CRC handbook
    'methanol': 0.791,  # CRC handbook
    'hexane': 0.655,  # CRC handbook
    'thf': 0.889,  # CRC handbook
    'toluene': 0.867,  # CRC handbook
    'acetic acid': 1.05,  # CRC handbook
    '1-butanol': 0.81,  # CRC handbook
    '2-butanol': 0.808,  # CRC handbook
    'ethanol': 0.789,  # CRC handbook
    'heptane': 0.684,  # CRC handbook
    'pentane': 0.626,  # CRC handbook
    '1-propanol': 0.803,  # CRC handbook
    'pyridine': 0.982,  # CRC handbook
    'ethyl acetate': 0.902,  # CRC handbook
    'cyclohexane': 0.779,  # CRC handbook
    'carbon tetrachloride': 1.59,  # CRC handbook
    'chlorobenzene': 1.11,  # CRC handbook
    '1,2-dichlorobenzene': 1.30,  # CRC handbook
    'n,n-dimethylacetamide': 0.937,  # CRC handbook
    'dioxane': 1.03,  # CRC handbook
    '1,2-ethanediol': 1.11,  # CRC handbook
    'decane': 0.73,  # CRC handbook
    'dibromomethane': 2.50,  # CRC handbook
    'dibutylether': 0.764,  # CRC handbook
    '1-bromopropane': 1.35,  # CRC handbook
    '2-bromopropane': 1.31,  # CRC handbook
    '1-chlorohexane': 0.88,  # CRC handbook
    '1-chloropentane': 0.88,  # CRC handbook
    '1-chloropropane': 0.89,  # CRC handbook
    'diethylamine': 0.707,  # CRC handbook
    '1-decanol': 0.83,  # CRC handbook
    'diiodomethane': 3.33,  # CRC handbook
    '1-fluorooctane': 0.88,  # CRC handbook
    '1-heptanol': 0.82,  # CRC handbook
    '1-hexanol': 0.814,  # CRC handbook
    '1-hexene': 0.673,  # CRC handbook
    '1-hexyne': 0.715,  # CRC handbook
    '1-iodobutane': 1.62,  # CRC handbook
    '1-iodohexadecane': 1.26,  # CRC handbook
    '1-iodopentane': 1.52,  # CRC handbook
    '1-iodopropane': 1.75,  # CRC handbook
    'dipropylamine': 0.738,  # CRC handbook
    'n-dodecane': 0.75,  # CRC handbook
    '1-nitropropane': 1.00,  # CRC handbook
    'ethanethiol': 0.839,  # CRC handbook
    '1-nonanol': 0.83,  # CRC handbook
    '1-octanol': 0.83,  # CRC handbook
    '1-pentanol': 0.814,  # CRC handbook
    '1-pentene': 0.64,  # CRC handbook
    'ethyl benzene': 0.867,  # CRC handbook
    '2,2,2-trifluoroethanol': 1.39,  # CRC handbook
    'fluorobenzene': 1.02,  # CRC handbook
    '2,2,4-trimethylpentane': 0.69,  # CRC handbook
    'formamide': 1.13,  # CRC handbook
    '2,4-dimethylpentane': 0.67,  # CRC handbook
    '2,4-dimethylpyridine': 0.93,  # CRC handbook
    '2,6-dimethylpyridine': 0.93,  # CRC handbook
    'n-hexadecane': 0.77,  # CRC handbook
    'dimethyl disulfide': 1.06,  # CRC handbook
    'ethyl methanoate': 0.92,  # CRC handbook
    'ethyl phenyl ether': 0.97,  # CRC handbook
    'formic acid': 1.22,  # CRC handbook
    'hexanoic acid': 0.93,  # CRC handbook
    '2-chlorobutane': 0.87,  # CRC handbook
    '2-heptanone': 0.81,  # CRC handbook
    '2-hexanone': 0.81,  # CRC handbook
    '2-methoxyethanol': 0.96,  # CRC handbook
    '2-methyl-1-propanol': 0.80,  # CRC handbook
    '2-methyl-2-propanol': 0.79,  # CRC handbook
    '2-methylpentane': 0.65,  # CRC handbook
    '2-methylpyridine': 0.95,  # CRC handbook
    '2-nitropropane': 1.00,  # CRC handbook
    '2-octanone': 0.82,  # CRC handbook
    '2-pentanone': 0.81,  # CRC handbook
    'iodobenzene': 1.83,  # CRC handbook
    'iodoethane': 1.93,  # CRC handbook
    'iodomethane': 2.28,  # CRC handbook
    'isopropylbenzene': 0.86,  # CRC handbook
    'p-isopropyltoluene': 0.86,  # CRC handbook
    'mesitylene': 0.86,  # CRC handbook
    'methyl benzoate': 1.09,  # CRC handbook
    'methyl butanoate': 0.90,  # CRC handbook
    'methyl ethanoate': 0.93,  # CRC handbook
    'methyl methanoate': 0.97,  # CRC handbook
    'methyl propanoate': 0.91,  # CRC handbook
    'n-methylaniline': 0.99,  # CRC handbook
    'methylcyclohexane': 0.77,  # CRC handbook
    'n-methylformamide (e/z mixture)': 1.01,  # CRC handbook
    'nitrobenzene': 1.20,  # CRC handbook
    'nitroethane': 1.05,  # CRC handbook
    'nitromethane': 1.14,  # CRC handbook
    'o-nitrotoluene': 1.16,  # CRC handbook
    'n-nonane': 0.72,  # CRC handbook
    'n-octane': 0.70,  # CRC handbook
    'n-pentadecane': 0.77,  # CRC handbook
    'pentanal': 0.81,  # CRC handbook
    'pentanoic acid': 0.94,  # CRC handbook
    'pentyl ethanoate': 0.88,  # CRC handbook
    'pentyl amine': 0.74,  # CRC handbook
    'perfluorobenzene': 1.61,  # CRC handbook
    'propanal': 0.81,  # CRC handbook
    'propanoic acid': 0.99,  # CRC handbook
    'propanenitrile': 0.78,  # CRC handbook
    'propyl ethanoate': 0.89,  # CRC handbook
    'propyl amine': 0.72,  # CRC handbook
    'tetrachloroethene': 1.62,  # CRC handbook
    'tetrahydrothiophene-s,s-dioxide': 1.26,  # CRC handbook
    'tetralin': 0.97,  # CRC handbook
    'thiophene': 1.06,  # CRC handbook
    'thiophenol': 1.07,  # CRC handbook
    'tributylphosphate': 0.98,  # CRC handbook
    'trichloroethene': 1.46,  # CRC handbook
    'triethylamine': 0.73,  # CRC handbook
    'n-undecane': 0.74,  # CRC handbook
    'xylene mixture': 0.86,  # CRC handbook
    'm-xylene': 0.86,  # CRC handbook
    'o-xylene': 0.88,  # CRC handbook
    'p-xylene': 0.86,  # CRC handbook
    '2-propanol': 0.785,  # CRC handbook
    '2-propen-1-ol': 0.85,  # CRC handbook
    'e-2-pentene': 0.65,  # CRC handbook
    '3-methylpyridine': 0.96,  # CRC handbook
    '3-pentanone': 0.81,  # CRC handbook
    '4-heptanone': 0.81,  # CRC handbook
    '4-methyl-2-pentanone': 0.80,  # CRC handbook
    '4-methylpyridine': 0.96,  # CRC handbook
    '5-nonanone': 0.82,  # CRC handbook
    'benzyl alcohol': 1.04,  # CRC handbook
    'butanoic acid': 0.96,  # CRC handbook
    'butanenitrile': 0.80,  # CRC handbook
    'butyl ethanoate': 0.88,  # CRC handbook
    'butylamine': 0.74,  # CRC handbook
    'n-butylbenzene': 0.86,  # CRC handbook
    'sec-butylbenzene': 0.86,  # CRC handbook
    'tert-butylbenzene': 0.86,  # CRC handbook
    'o-chlorotoluene': 1.08,  # CRC handbook
    'm-cresol': 1.03,  # CRC handbook
    'o-cresol': 1.07,  # CRC handbook
    'cyclohexanone': 0.95,  # CRC handbook
    'isoquinoline': 1.10,  # CRC handbook
    'quinoline': 1.09,  # CRC handbook
    'argon': 0.001784,  # CRC handbook
    'krypton': 0.003733,  # CRC handbook
    'xenon': 0.005894,  # CRC handbook
}
