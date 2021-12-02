import numpy as np
from typing import Union, Sequence, List
from scipy.spatial.distance import cdist
from scipy.stats import special_ortho_group
from mltrain.configurations import Configuration
from mltrain.log import logger
from mltrain.box import Box
from mltrain.molecule import Molecule


class System:
    """System with molecules but without any coordinates"""

    def __init__(self,
                 *args: Molecule,
                 box:   Union[Box, Sequence[float], None]):
        """
        System containing a set of molecules.

        e.g. pd_1water = (Pd, water, box_size=[10, 10, 10], charge=2)
        for a system containing a Pd(II) ion and one water in a 1 nm^3 box

        -----------------------------------------------------------------------
        Arguments:
            args: Molecules that comprise the system

            box: Box that the molecules occupy. e.g. [10, 10, 10] for a
                 10 Å cubic box.
        """
        self.molecules = list(args)

        if box is None:
            logger.info('System is in the gas phase')
            self.box = None

        else:
            self.box = box if isinstance(box, Box) else Box(box)

    def random_configuration(self,
                             min_dist:     float = 2.0,
                             with_intra:   bool = False,
                             intra_sigma:  float = 0.01
                             ) -> 'mltrain.Configuration':
        """
        Generate a random configuration of this system, where all the molecules
        in the system have been randomised

        -----------------------------------------------------------------------
        Arguments:
            min_dist: Minimum distance between the molecules

            with_intra: Add intramolecular displacements to the atoms

            intra_sigma: Variance of the normal distribution used to displace
                         each atom_pair_list of the molecules in the system

        Returns:
            (mltrain.Configuration):

        Raises:
            (RuntimeError): If all the molecules cannot be randomised while
                            maintaining the required min. distance between them
        """
        configuration = Configuration(charge=self.charge,
                                      mult=self.mult)

        for molecule in self.molecules:

            if with_intra:
                logger.info(f'Adding random normal displacement with '
                            f'σ={intra_sigma} Å')
                molecule.random_normal_jiggle(sigma=intra_sigma)

            self._shift_to_midpoint(molecule)
            if configuration.n_atoms > 0:
                self._rotate_randomly(molecule)
                self._shift_randomly(molecule,
                                     coords=configuration.coordinates,
                                     min_dist=min_dist)

            configuration.atoms += molecule.atoms.copy()

        return configuration

    @property
    def configuration(self) -> 'mltrain.Configuration':
        """
        Single configuration for this system

        -----------------------------------------------------------------------
        Returns:
            (mltrain.Configuration):
        """

        if len(self.molecules) == 1:
            return self.random_configuration(with_intra=False)

        else:
            raise NotImplementedError("A single configuration for a system "
                                      "with > 1 molecule(s) is not implemented"
                                      " Call random_configuration()")

    def add_molecule(self,
                     molecule: 'mltrain.Molecule'
                     ) -> None:
        """
        Add a molecule to this system

        -----------------------------------------------------------------------
        Arguments:
            molecule:
        """

        self.molecules.append(molecule)
        return None

    def add_molecules(self,
                      molecule: 'mltrain.Molecule',
                      num:      int = 1
                      ):
        """
        Add multiple versions of a molecule to this sytem

        -----------------------------------------------------------------------
        Arguments:
            molecule:

        Keyword Arguments:
            num: Number of molecules of this type to add
        """

        for _ in range(num):
            self.add_molecule(molecule.copy())

        return None

    @property
    def charge(self) -> int:
        """Get the total charge on the system"""
        return sum(molecule.charge for molecule in self.molecules)

    @property
    def mult(self) -> int:
        """Get the total spin multiplicity on the system"""
        n_unpaired = sum((mol.mult - 1) / 2 for mol in self.molecules)
        return 2 * n_unpaired + 1

    @property
    def atoms(self) -> List['autode.atoms.Atom']:
        """Constituent atoms of this system

        -----------------------------------------------------------------------
        Returns:
            (list(autode.atoms.Atom)):
        """
        return sum((mol.atoms for mol in self.molecules), None)

    @property
    def unique_atomic_symbols(self) -> List[str]:
        """
        Unique atomic symbols in this system

        -----------------------------------------------------------------------
        Returns:
            (list(str)):
        """
        return list(sorted(set([a.label for a in self.atoms])))

    def _shift_to_midpoint(self, molecule) -> None:
        """Shift a molecule to the midpoint in the box, if defined"""
        midpoint = np.zeros(3) if self.box is None else self.box.midpoint
        molecule.translate(midpoint - molecule.centroid)
        return None

    @staticmethod
    def _rotate_randomly(molecule) -> None:
        """Rotate a molecule randomly around it's centroid"""
        logger.info(f'Rotating {molecule.name} about its centroid')

        coords, centroid = molecule.coordinates, molecule.centroid

        #                Shift to origin     Random rotation matrix
        coords = np.dot(coords - centroid, special_ortho_group.rvs(3).T)
        molecule.coordinates = coords + centroid

        return None

    def _shift_randomly(self, molecule, coords, min_dist, max_iters=500) -> None:
        """
        Shift a molecule such that that there more than min_dist between
        each of a molecule's coordinates and a current set
        """
        logger.info(f'Shifting {molecule.name} to a random position in a box')

        molecule_coords = molecule.coordinates
        a, b, c = [1.0, 1.0, 1.0] if self.box is None else self.box.size

        def too_close(_coords) -> bool:
            """Are a set of coordinates too close to some others?"""
            return np.min(cdist(_coords, coords)) < min_dist

        def in_box(_coords) -> bool:
            """Are a set of coordinates all inside a box?"""
            if self.box is None:
                return True

            max_delta = np.max(np.max(_coords, axis=0) - np.array([a, b, c]))
            return np.min(_coords) > 0.0 and max_delta < 0

        for i in range(1, max_iters+1):

            m_coords = np.copy(molecule_coords)
            vec = [np.random.uniform(-a/2, a/2),    # Random translation vector
                   np.random.uniform(-b/2, b/2),
                   np.random.uniform(-c/2, c/2)]

            # Shift by 0.1 increments in the random direction
            vec = 0.1 * np.array(vec) / np.linalg.norm(vec)

            while too_close(m_coords) and in_box(m_coords):
                m_coords += vec

            if not too_close(m_coords) and in_box(m_coords):
                logger.info(f'Randomised in {i} iterations')
                break

            if i == max_iters:
                raise RuntimeError(f'Failed to shift {molecule.formula} to a '
                                   f'random location in the box. '
                                   f'Tried {max_iters} times')

        molecule.coordinates = m_coords
        return
