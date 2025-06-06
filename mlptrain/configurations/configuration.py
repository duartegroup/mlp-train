import mlptrain
import ase
import numpy as np
import os
import json
from typing import Optional, Union, List, Dict
from copy import deepcopy
from autode.atoms import AtomCollection, Atom
import autode.atoms
import ase.atoms
from mlptrain.log import logger
from mlptrain.energy import Energy
from mlptrain.forces import Forces
from mlptrain.box import Box
from mlptrain.configurations.calculate import run_autode
from mlptrain.utils import work_in_tmp_dir
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

        # Dictionary to track molecule types and their atom ranges
        self.mol_dict: Dict[str, List[Dict[str, Union[int, str]]]] = {}

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
        random_seed: int = 42,
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

        random_seed: int = 42
            The seed number used to generate random vectors for the rotation and translation of the solvent molecules.
            This is to avoid the same solvent molecule being inserted multiple times in the same orientation.

        ___________________________________________________________________________

        """
        # Calculate the box size if not provided, based on the maximum distance between any two atoms in the solute
        # and the buffer distance
        if box_size is None:
            box_size = _get_max_mol_distance(self.atoms) + buffer_distance

        # Assume cubic box
        self.box = Box([box_size] * 3)

        if None not in (solvent_density, solvent_molecule, solvent_name):
            raise ValueError(
                'Either the solvent name or the combination of solvent molecule and density must be provided.'
                'You shoult not provide all three.'
            )

        # If both solvent molecule and density are provided, stop checking
        elif None not in (solvent_density, solvent_molecule):
            if solvent_molecule.atoms is None:
                raise ValueError('The solvent molecule must contain atoms')
            if solvent_density <= 0:
                raise ValueError(
                    'The density of the solvent must be greater than 0'
                )

        # If the solvent name is provided, get the solvent molecule and density from the solvent database
        # by getting the smiles from autode's solvent database, creating a molecule object and optimising it
        # with xtb, then get the density from the density database
        elif solvent_name is not None:
            logger.info(
                f'Searching solvent with the name {solvent_name} in autodE solvent database'
            )
            solvent = get_solvent(solvent_name, kind='implicit')
            solvent_smiles = solvent.smiles
            solvent_molecule = ade.Molecule(smiles=solvent_smiles)
            solvent_molecule = optimise_solvent(solvent_molecule)

            if solvent.name not in solvent_densities.keys():
                raise ValueError(
                    f'The density of {solvent.name} is not in the database'
                    f'Please provide the solvent molecule and density explicitly'
                )
            else:
                solvent_density = solvent_densities[solvent.name]
                logger.info(
                    f'Found solvent {solvent.name} with density {solvent_density} g/cm^3'
                )

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
            solvent_molecule,
            box_size,
            contact_threshold,
            solvent_number,
            random_seed,
        )

    def k_d_tree_insertion(
        self,
        solvent_molecule: ade.Molecule,
        box_size: float,
        contact_threshold: float,
        n_solvent: int,
        random_seed: int,
    ) -> np.ndarray:
        """Insert solvent molecules into the box using a k-d tree to check for collisions.
        Implemented according to the algorithm described in the paper:

        "https://chemrxiv.org/engage/chemrxiv/article-details/678621ccfa469535b9ea8786"

        This implementation includes periodic boundary condition handling by creating
        periodic images of atoms near box boundaries.

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

        # Initialize mol_dict with the original solute molecule
        if not self.mol_dict:
            self.mol_dict['solute'] = [
                {
                    'start': 0,
                    'end': len(system_coords),
                    'formula': self._get_formula_from_atoms(self.atoms),
                }
            ]

        # Get solvent name for mol_dict (use formula as fallback)
        solvent_name = getattr(
            solvent_molecule, 'name', solvent_molecule.formula
        )
        if solvent_name not in self.mol_dict:
            self.mol_dict[solvent_name] = []

        # Initialise a count of the solvents inserted into the box. The ideal solvent number is
        # the number of solvent molecules that fit into the box, but this number might not be reached
        # because the solute might be in the way of some of the solvent molecules
        solvents_inserted = 0
        seeded_random = random.Random(random_seed)

        for i in range(n_solvent):
            # Create periodic images for boundary condition handling
            periodic_coords = _create_periodic_images(
                system_coords, box_size, contact_threshold
            )

            # Build a k-d tree from the system coordinates including periodic images
            existing_tree = _build_cKDTree(periodic_coords)
            inserted = False
            attempt = 0

            # Try to insert the solvent molecule into the box with a maximum of 1000 attempts
            while not inserted and attempt < 1000:
                attempt += 1

                # Generate a random rotation matrix and a random translation vector which
                # depends on a seed number that is the product of the current iteration (number of solvent
                # being added) to the third power and the attempt number. This is to avoid
                # the same solvent molecule being inserted multiple times in the same orientation
                rot_matrix = _random_rotation(
                    seeded_random.random(),
                    seeded_random.random(),
                    seeded_random.random(),
                )
                rot_solvent = np.dot(solvent_coords, rot_matrix)
                translation = _random_vector_in_box(
                    box_size,
                    seeded_random.random(),
                    seeded_random.random(),
                    seeded_random.random(),
                )

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
                # This now includes periodic boundary conditions through the periodic images
                distances, indeces = existing_tree.query(trial_coords)
                if all(distances > contact_threshold):
                    solvent_translated = deepcopy(solvent_molecule)

                    # Record the starting position of this molecule
                    start_index = len(self.atoms)

                    for n, atom in enumerate(solvent_translated.atoms):
                        atom.coordinate = trial_coords[n]

                    self.atoms.extend(solvent_translated.atoms)
                    system_coords = np.concatenate(
                        (system_coords, trial_coords)
                    )

                    # Add molecule info to mol_dict
                    end_index = len(self.atoms)
                    self.mol_dict[solvent_name].append(
                        {
                            'start': start_index,
                            'end': end_index,
                            'formula': solvent_molecule.formula,
                        }
                    )

                    inserted = True
                    solvents_inserted += 1

        logger.info(
            f'Inserted {solvents_inserted} solvent molecules into the box'
        )

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

        # Automatically save mol_dict if it exists
        if self.mol_dict:
            self.save_mol_dict(filename)

        return None

    def save_mol_dict(self, filename: str) -> None:
        """
        Save the mol_dict to a hidden .mol_dict.txt file alongside the xyz file.

        Arguments:
            filename: The xyz filename (mol_dict will be saved as .filename.mol_dict.txt)
        """
        if not self.mol_dict:
            return

        # Create the hidden mol_dict filename
        directory = os.path.dirname(filename) or '.'
        base_name = os.path.splitext(os.path.basename(filename))[0]
        mol_dict_file = os.path.join(directory, f'.{base_name}.mol_dict.txt')

        # Save mol_dict as JSON for easy reading/writing
        with open(mol_dict_file, 'w') as f:
            json.dump(self.mol_dict, f, indent=2)

        logger.info(f'Saved mol_dict to {mol_dict_file}')

    def load_mol_dict(self, filename: str) -> bool:
        """
        Load mol_dict from a hidden .mol_dict.txt file if it exists.

        Arguments:
            filename: The xyz filename to check for an associated mol_dict file

        Returns:
            bool: True if mol_dict was loaded, False if file doesn't exist
        """

        # Create the hidden mol_dict filename
        directory = os.path.dirname(filename) or '.'
        base_name = os.path.splitext(os.path.basename(filename))[0]
        mol_dict_file = os.path.join(directory, f'.{base_name}.mol_dict.txt')

        if os.path.exists(mol_dict_file):
            try:
                with open(mol_dict_file, 'r') as f:
                    self.mol_dict = json.load(f)
                logger.info(f'Loaded mol_dict from {mol_dict_file}')
                return True
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f'Failed to load mol_dict from {mol_dict_file}: {e}'
                )
                return False
        return False

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

    def _get_formula_from_atoms(self, atoms) -> str:
        """
        Generate a molecular formula from a list of atoms.

        Arguments:
            atoms: List of atoms

        Returns:
            str: Molecular formula (e.g., "H2O", "C2H6O")
        """
        from collections import Counter

        # Count atoms by element
        element_counts = Counter(atom.label for atom in atoms)

        # Build formula string
        formula_parts = []
        for element in sorted(element_counts.keys()):
            count = element_counts[element]
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f'{element}{count}')

        return ''.join(formula_parts)

    @classmethod
    def from_xyz(
        cls, filename: str, charge: int = 0, mult: int = 1
    ) -> 'Configuration':
        """
        Create a Configuration from an xyz file and automatically load mol_dict if available.

        Arguments:
            filename: Path to xyz file
            charge: Overall charge
            mult: Spin multiplicity

        Returns:
            Configuration: New configuration with mol_dict loaded if available
        """
        import ase.io

        # Load atoms from xyz file
        ase_atoms = ase.io.read(filename)

        # Convert to autode atoms
        from autode.atoms import Atom

        atoms = [
            Atom(symbol, x=coord[0], y=coord[1], z=coord[2])
            for symbol, coord in zip(
                ase_atoms.get_chemical_symbols(), ase_atoms.get_positions()
            )
        ]

        # Create box if cell information is available
        box = None
        cell_array = np.array(ase_atoms.get_cell())
        if np.any(cell_array != 0):
            from mlptrain.box import Box

            box = Box(np.diag(cell_array))

        # Create configuration
        config = cls(atoms=atoms, charge=charge, mult=mult, box=box)

        # Try to load mol_dict
        config.load_mol_dict(filename)

        return config

    def validate_mol_dict(self) -> bool:
        """
        Validate that the mol_dict indices are consistent with the current atoms.

        Returns:
            bool: True if mol_dict is valid, False otherwise
        """
        if not self.mol_dict:
            return True

        total_atoms = len(self.atoms)

        for mol_type, molecules in self.mol_dict.items():
            for i, mol_info in enumerate(molecules):
                start = mol_info.get('start', 0)
                end = mol_info.get('end', 0)

                # Check bounds
                if start < 0 or end > total_atoms or start >= end:
                    logger.warning(
                        f'Invalid mol_dict entry: {mol_type}[{i}] has start={start}, end={end}, but total atoms={total_atoms}'
                    )
                    return False

        return True


def _random_rotation(r1: float, r2: float, r3: float) -> np.ndarray:
    """Generate a random rotation matrix"""
    theta = r1 * 360
    kappa = r2 * 360
    gamma = r3 * 360

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


def _random_vector_in_box(
    box_size: float,
    r1: float,
    r2: float,
    r3: float,
) -> np.ndarray:
    """Generate a random vector in a box"""
    return np.array([r * box_size for r in [r1, r2, r3]])


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


@work_in_tmp_dir()
def optimise_solvent(solvent: ade.Molecule) -> ade.Molecule:
    """Optimise a solvent molecule with XTB"""
    solvent_copy = deepcopy(solvent)
    solvent_copy.optimise(method=ade.methods.XTB())
    return solvent_copy


def _create_periodic_images(
    coords: np.ndarray, box_size: float, contact_threshold: float
) -> np.ndarray:
    """
    Create periodic images of atoms that are within contact_threshold distance
    of the box boundaries to handle periodic boundary conditions.

    Arguments:
        coords: Original coordinates of atoms in the box
        box_size: Size of the cubic box
        contact_threshold: Distance threshold for considering periodic images

    Returns:
        Combined array of original coordinates and their periodic images
    """
    original_coords = coords.copy()
    periodic_images = []

    # For each atom, check if it's close to any boundary
    for coord in coords:
        x, y, z = coord

        # Check each dimension for proximity to boundaries
        # Create images for atoms within contact_threshold of each face
        images_to_add = []

        # X-direction boundaries
        if x < contact_threshold:  # Close to x=0 face
            images_to_add.append([x + box_size, y, z])
        if x > (box_size - contact_threshold):  # Close to x=box_size face
            images_to_add.append([x - box_size, y, z])

        # Y-direction boundaries
        if y < contact_threshold:  # Close to y=0 face
            images_to_add.append([x, y + box_size, z])
        if y > (box_size - contact_threshold):  # Close to y=box_size face
            images_to_add.append([x, y - box_size, z])

        # Z-direction boundaries
        if z < contact_threshold:  # Close to z=0 face
            images_to_add.append([x, y, z + box_size])
        if z > (box_size - contact_threshold):  # Close to z=box_size face
            images_to_add.append([x, y, z - box_size])

        # Add corner and edge images for atoms close to multiple boundaries
        # X-Y corners
        if x < contact_threshold and y < contact_threshold:
            images_to_add.append([x + box_size, y + box_size, z])
        if x < contact_threshold and y > (box_size - contact_threshold):
            images_to_add.append([x + box_size, y - box_size, z])
        if x > (box_size - contact_threshold) and y < contact_threshold:
            images_to_add.append([x - box_size, y + box_size, z])
        if x > (box_size - contact_threshold) and y > (
            box_size - contact_threshold
        ):
            images_to_add.append([x - box_size, y - box_size, z])

        # X-Z corners
        if x < contact_threshold and z < contact_threshold:
            images_to_add.append([x + box_size, y, z + box_size])
        if x < contact_threshold and z > (box_size - contact_threshold):
            images_to_add.append([x + box_size, y, z - box_size])
        if x > (box_size - contact_threshold) and z < contact_threshold:
            images_to_add.append([x - box_size, y, z + box_size])
        if x > (box_size - contact_threshold) and z > (
            box_size - contact_threshold
        ):
            images_to_add.append([x - box_size, y, z - box_size])

        # Y-Z corners
        if y < contact_threshold and z < contact_threshold:
            images_to_add.append([x, y + box_size, z + box_size])
        if y < contact_threshold and z > (box_size - contact_threshold):
            images_to_add.append([x, y + box_size, z - box_size])
        if y > (box_size - contact_threshold) and z < contact_threshold:
            images_to_add.append([x, y - box_size, z + box_size])
        if y > (box_size - contact_threshold) and z > (
            box_size - contact_threshold
        ):
            images_to_add.append([x, y - box_size, z - box_size])

        # 3D corners (8 corner cases)
        if (
            x < contact_threshold
            and y < contact_threshold
            and z < contact_threshold
        ):
            images_to_add.append([x + box_size, y + box_size, z + box_size])
        if (
            x < contact_threshold
            and y < contact_threshold
            and z > (box_size - contact_threshold)
        ):
            images_to_add.append([x + box_size, y + box_size, z - box_size])
        if (
            x < contact_threshold
            and y > (box_size - contact_threshold)
            and z < contact_threshold
        ):
            images_to_add.append([x + box_size, y - box_size, z + box_size])
        if (
            x < contact_threshold
            and y > (box_size - contact_threshold)
            and z > (box_size - contact_threshold)
        ):
            images_to_add.append([x + box_size, y - box_size, z - box_size])
        if (
            x > (box_size - contact_threshold)
            and y < contact_threshold
            and z < contact_threshold
        ):
            images_to_add.append([x - box_size, y + box_size, z + box_size])
        if (
            x > (box_size - contact_threshold)
            and y < contact_threshold
            and z > (box_size - contact_threshold)
        ):
            images_to_add.append([x - box_size, y + box_size, z - box_size])
        if (
            x > (box_size - contact_threshold)
            and y > (box_size - contact_threshold)
            and z < contact_threshold
        ):
            images_to_add.append([x - box_size, y - box_size, z + box_size])
        if (
            x > (box_size - contact_threshold)
            and y > (box_size - contact_threshold)
            and z > (box_size - contact_threshold)
        ):
            images_to_add.append([x - box_size, y - box_size, z - box_size])

        periodic_images.extend(images_to_add)

    # Combine original coordinates with periodic images
    if periodic_images:
        periodic_images = np.array(periodic_images)
        return np.vstack([original_coords, periodic_images])
    else:
        return original_coords


# Solvent densities were taken from the CRC handbook
# Temperatures at which the densities were measured are also provided
# The densities are in g/cm^3, temperatures in K
# CRC Handbook of Chemistry and Physics, 97th ed.; Haynes, W. M., Ed.;
# CRC Press: Boca Raton, FL, 2016; ISBN 978-1-4987-5429-3.
solvent_densities = {
    'water': 1.0,  # 298K
    'dichloromethane': 1.33,  # 293K
    'acetone': 0.79,  # 293K
    'acetonitrile': 0.786,  # 293K
    'benzene': 0.874,  # 293K
    'trichloromethane': 1.49,  # 293K
    'cs2': 1.26,  # 293K
    'dmf': 0.944,  # 298K
    'dmso': 1.10,  # 298K
    'diethyl ether': 0.713,  # 298K
    'methanol': 0.791,  # 293K
    'hexane': 0.655,  # 293K
    'thf': 0.889,  # 298K
    'toluene': 0.867,  # 293
    'acetic acid': 1.05,  # 293K
    '1-butanol': 0.81,  # 293K
    '2-butanol': 0.808,  # 293K
    'ethanol': 0.789,  # 293K
    'heptane': 0.684,  # 293K
    'pentane': 0.626,  # 293K
    '1-propanol': 0.803,  # 293K
    'pyridine': 0.982,  # 293K
    'ethyl acetate': 0.902,  # 293K
    'cyclohexane': 0.779,  # 293K
    'carbon tetrachloride': 1.59,  # 293K
    'chlorobenzene': 1.11,  # 293K
    '1,2-dichlorobenzene': 1.30,  # 293K
    'n,n-dimethylacetamide': 0.937,  # 298K
    'dioxane': 1.03,  # 293K
    '1,2-ethanediol': 1.11,  # 293K
    'decane': 0.73,  # 293K
    'dibromomethane': 2.50,  # 293K
    'dibutylether': 0.764,  # 293K
    '1-bromopropane': 1.35,  # 293K
    '2-bromopropane': 1.31,  # 293K
    '1-chloropentane': 0.88,  # 293K
    '1-chloropropane': 0.89,  # 293K
    'diethylamine': 0.707,  # 293K
    '1-decanol': 0.83,  # 293K
    'diiodomethane': 3.33,  # 293K
    '1-heptanol': 0.82,  # 293K
    '1-hexanol': 0.814,  # 293K
    '1-hexene': 0.667,  # 298K
    '1-iodopropane': 1.75,  # 293K
    'dipropylamine': 0.74,  # 293K
    'n-dodecane': 0.75,  # 293K
    '1-nitropropane': 1.00,  # 298K
    'ethanethiol': 0.83,  # 298K
    '1-nonanol': 0.83,  # 293K
    '1-octanol': 0.83,  # 293K
    '1-pentanol': 0.814,  # 293K
    '1-pentene': 0.64,  # 293K
    'ethyl benzene': 0.87,  # 293K
    'fluorobenzene': 1.02,  # 293K
    '2,2,4-trimethylpentane': 0.69,  # 293K
    'formamide': 1.13,  # 293K
    '2,6-dimethylpyridine': 0.93,  # 293K
    'dimethyl disulfide': 0.85,  # 293K
    'formic acid': 1.22,  # 293K
    'hexanoic acid': 0.93,  # 293K
    '2-chlorobutane': 0.87,  # 293K
    '2-heptanone': 0.81,  # 293K
    '2-hexanone': 0.81,  # 293K
    '2-methoxyethanol': 0.96,  # 293K
    '2-methyl-1-propanol': 0.80,  # 293K
    '2-methyl-2-propanol': 0.79,  # 293K
    '2-methylpentane': 0.65,  # 298K
    '2-methylpyridine': 0.94,  # 293K
    '2-nitropropane': 0.98,  # 298K
    '2-octanone': 0.82,  # 293K
    '2-pentanone': 0.81,  # 293K
    'iodobenzene': 1.83,  # 293K
    'iodoethane': 1.94,  # 293K
    'iodomethane': 2.28,  # 293K
    'isopropylbenzene': 0.86,  # 293K
    'methyl benzoate': 1.09,  # 298K
    'methyl butanoate': 0.90,  # 293K
    'methyl ethanoate': 0.93,  # 293K
    'methyl methanoate': 0.97,  # 293K
    'methyl propanoate': 0.92,  # 293K
    'n-methylaniline': 0.99,  # 293K
    'methylcyclohexane': 0.77,  # 293K
    'n-methylformamide (e/z mixture)': 1.01,  # 293K
    'nitrobenzene': 1.20,  # 292K
    'nitroethane': 1.05,  # 298K
    'nitromethane': 1.14,  # 293K
    'o-nitrotoluene': 1.16,  # 292K
    'n-nonane': 0.72,  # 293K
    'n-octane': 0.70,  # 293K
    'pentanal': 0.81,  # 293K
    'pentanoic acid': 0.94,  # 293K
    'pentyl ethanoate': 0.88,  # 293K
    'pentyl amine': 0.75,  # 293K
    'perfluorobenzene': 1.61,  # 293K
    'propanal': 0.87,  # 298K
    'propanoic acid': 0.99,  # 298K
    'propanenitrile': 0.78,  # 298K
    'propyl ethanoate': 0.89,  # 293K
    'propyl amine': 0.72,  # 293K
    'tetrachloroethene': 1.62,  # 293K
    'thiophene': 1.06,  # 293K
    'thiophenol': 1.07,  # 293K
    'trichloroethene': 1.46,  # 293K
    'triethylamine': 0.73,  # 293K
    'n-undecane': 0.74,  # 293K
    'm-xylene': 0.86,  # 293K
    'o-xylene': 0.88,  # 293K
    'p-xylene': 0.86,  # 293K
    '2-propanol': 0.79,  # 293K
    'e-2-pentene': 0.65,  # 298K
    '3-methylpyridine': 0.96,  # 293K
    '3-pentanone': 0.81,  # 298
    '4-heptanone': 0.81,  # 293K
    '4-methyl-2-pentanone': 0.80,  # 298
    '4-methylpyridine': 0.96,  # 293
    'benzyl alcohol': 1.04,  # 293
    'butanoic acid': 0.96,  # 298
    'butyl ethanoate': 0.88,  # 293
    'butylamine': 0.74,  # 293
    'n-butylbenzene': 0.86,  # 293
    'tert-butylbenzene': 0.87,  # 293
    'o-chlorotoluene': 1.08,  # 293
    'm-cresol': 1.03,  # 293
    'o-cresol': 1.03,  # 308K
    'cyclohexanone': 0.95,  # 293
    'isoquinoline': 1.10,  # 303K
    'quinoline': 1.09,  # 288K
}
