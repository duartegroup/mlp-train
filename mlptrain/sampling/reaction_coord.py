import numpy as np
import ase
import mlptrain as mlt
from abc import ABC, abstractmethod
from typing import Union
from mlptrain.sampling._base import Function


class ReactionCoordinate(Function, ABC):

    def __call__(self,
                 arg: Union[ase.atoms.Atoms,
                            'mlptrain.Configuration',
                            'mlptrain.ConfigurationSet']
                 ) -> Union[float, np.ndarray]:
        """Value of this reaction coordinate"""

        if isinstance(arg, ase.atoms.Atoms):
            return self._call(arg)

        elif isinstance(arg, mlt.Configuration):
            return self._call(arg.ase_atoms)

        elif isinstance(arg, mlt.ConfigurationSet):
            return np.array([self._call(c.ase_atoms) for c in arg])

        else:
            raise ValueError('Reaction coordinate must be called using ase '
                             'atoms, a configuration or configuration set')

    @abstractmethod
    def _call(self, atoms: ase.atoms.Atoms):
        """Evaluate this function for a set of ase atoms"""

    def grad(self, atoms: ase.atoms.Atoms):
        """Gradient of this reaction coordinate for a set of ase atoms"""

        if not isinstance(atoms, ase.atoms.Atoms):
            raise NotImplementedError('Grad must be called with a set of '
                                      'ASE atoms')

        return self._grad(atoms)

    @abstractmethod
    def _grad(self, atoms: ase.atoms.Atoms):
        """Gradient for a set of ASE atoms"""


class DummyCoordinate(ReactionCoordinate):

    def _call(self, atoms: ase.atoms.Atoms):
        raise ValueError('Cannot call energy on a dummy coordinate')

    def _grad(self, atoms: ase.atoms.Atoms):
        raise ValueError('Cannot call grad on a dummy coordinate')


class AverageDistance(ReactionCoordinate):
    """Average distance between each pair of atoms specified"""

    def __init__(self, *args):
        """
        Average of a set of distances e.g.

        # Average of a single distance between atoms 0 and 1
        dists = AverageDistance((0, 1))

        # or multiple distances (0-1 and 1-2)
        dists = AverageDistance((0, 1), (1, 2))

        -----------------------------------------------------------------------
        Arguments:
            args: Pairs of atom indices
        """

        self.atom_pair_list = []

        for arg in args:
            if len(arg) != 2:
                raise ValueError('Average distance must be initialised from '
                                 'a 2-tuple of atom indices')

            self.atom_pair_list.append(tuple(arg))

        atom_idxs = [idx for pair in self.atom_pair_list for idx in pair]

        if len(set(atom_idxs)) != len(atom_idxs):
            raise ValueError('All atoms in reaction coordinate must be '
                             'different')

    def _call(self, atoms: ase.atoms.Atoms):
        """Average distance between atom pairs"""
        return np.mean([atoms.get_distance(i, j, mic=True)
                        for (i, j) in self.atom_pair_list])

    def _grad(self, atoms: ase.atoms.Atoms):
        """Gradient of the average distance between atom pairs. Each component
        of the gradient is calculated using ∇B_i,m:

        ∇B_i,m = (1/M) * (r_i,m - r_i,m') / ||r_mm'||

        ∇B_i,m:  Gradient of bias for atom m along component i for pair m, m'
        M:       Number of atom pairs
        r_i,m:   i (= x, y or z) position of atom in pair m, m'
        ||r_m||: Euclidean distance between atoms in pair m, m'
        """

        derivative = np.zeros(shape=(len(atoms), 3))

        distances = [atoms.get_distance(i, j, mic=True)
                     for (i, j) in self.atom_pair_list]

        for m, (i, j) in enumerate(self.atom_pair_list):
            x_dist, y_dist, z_dist = [atoms[i].position[k] -
                                      atoms[j].position[k] for k in
                                      range(3)]

            x_i = x_dist / (self.n_pairs * distances[m])
            y_i = y_dist / (self.n_pairs * distances[m])
            z_i = z_dist / (self.n_pairs * distances[m])

            derivative[i][:] = [x_i, y_i, z_i]
            derivative[j][:] = [-x_i, -y_i, -z_i]

        return derivative

    @property
    def n_pairs(self) -> int:
        """Number of pairs of atoms in this set"""
        return len(self.atom_pair_list)


class DifferenceDistance(ReactionCoordinate):
    """Difference in distance between two pairs of specified atoms"""

    def __init__(self, *args):
        """
        Difference of a pair of distances e.g.

        # Δr = r_1 - r_2, where r_i is the distance between atoms j and k
        dists = DifferenceDistance((0, 1), (0, 2))

        Must comprise two pairs of atoms

        -----------------------------------------------------------------------
        Arguments:
            args: Pairs of atom indices
        """

        self.atom_pair_list = []

        if len(args) != 2:
            raise ValueError('Difference Distance must comprise exactly two '
                             'pairs of atoms')

        for arg in args:
            if len(arg) != 2:
                raise ValueError('Average distance must be initialised from '
                                 'a 2-tuple of atom indices')

            self.atom_pair_list.append(tuple(arg))

    def _call(self, atoms: ase.atoms.Atoms):
        """Difference in distance between two atom pairs"""
        dists = [atoms.get_distance(i, j, mic=True)
                 for (i, j) in self.atom_pair_list]

        return dists[0] - dists[1]

    def _grad(self, atoms: ase.atoms.Atoms):
        """Gradient of the difference in distance between two atom pairs. Each
        component of the gradient is calculated using ∇B_i,m:

        ∇B_i,m = (r_i,m - r_i,m') / ||r_mm'||

        An additional term is subtracted if there are only 3 atoms in the 2
        pairs. e.g. - (r_i,m - r_i,n) / ||r_mn||

        ∇B_i,m:  Gradient of bias for atom m along component i for pair m, m'
        r_i,m:   i (= x, y or z) position of atom in pair m, m'
        ||r_m||: Euclidean distance between atoms in pair m, m'
        """

        derivative = np.zeros(shape=(len(atoms), 3))

        distances = [atoms.get_distance(i, j, mic=True)
                     for (i, j) in self.atom_pair_list]

        for m, (i, j) in enumerate(self.atom_pair_list):

            x_dist, y_dist, z_dist = [atoms[i].position[k] -
                                      atoms[j].position[k] for k in
                                      range(3)]

            x_i = x_dist / distances[m]
            y_i = y_dist / distances[m]
            z_i = z_dist / distances[m]

            if m == 0:

                derivative[i][:] += [x_i, y_i, z_i]
                derivative[j][:] += [-x_i, -y_i, -z_i]

            elif m == 1:

                derivative[i][:] += [-x_i, -y_i, -z_i]
                derivative[j][:] += [x_i, y_i, z_i]

        return derivative
