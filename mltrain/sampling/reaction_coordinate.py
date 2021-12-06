import numpy as np
import ase
import mltrain as mlt
from abc import ABC, abstractmethod
from typing import Union
from mltrain.sampling._base import Function


class ReactionCoordinate(Function, ABC):

    def __call__(self,
                 arg: Union[ase.atoms.Atoms,
                            'mltrain.Configuration',
                            'mltrain.ConfigurationSet']
                 ) -> Union[float, np.ndarray]:
        """Value of this reaction coordinate"""

        if isinstance(arg, ase.atoms.Atoms):
            return self._call(arg)

        elif isinstance(arg, mlt.Configuration):
            return self._call(arg.ase_atoms)

        elif isinstance(arg, mlt.ConfigurationSet):
            return np.array([self._call(c.ase_atoms) for c in arg])

        else:
            raise ValueError('Reaciton coordinate must be called using ase '
                             'atoms a configuration or configuration set')

    @abstractmethod
    def _call(self, atoms: ase.atoms.Atoms):
        """Call this function on a set of con"""

    def grad(self, atoms: ase.atoms.Atoms):
        """Gradient of this reaction coordinate"""

        if not isinstance(atoms, ase.atoms.Atoms):
            raise NotImplementedError('Grad must be called with a set of '
                                      'ASE atoms')

        return self._grad(atoms)

    @abstractmethod
    def _grad(self, atoms: ase.atoms.Atoms):
        """Gradient for a set of ASE atoms"""


class AverageDistance(ReactionCoordinate):
    """Average distance between each pair of atoms specified"""

    def __init__(self, *args):
        """
        Average of a set of distances

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

    def _call(self, atoms: ase.atoms.Atoms):
        """Average distance between atom pairs"""
        return np.mean([atoms.get_distance(i, j, mic=True)
                        for (i, j) in self.atom_pair_list])

    def _grad(self, atoms: ase.atoms.Atoms):
        """Gradient of the average distance between atom pairs. Each component
        of the gradient is calculated using ∇B_i,m:

        ∇B_i,m = (1/M) * (r_i,m - r_i,m') / ||r_m||

        ∇B_i,m:  Gradient of bias for atom m along component i for pair m, m'
        M:       Number of atom pairs
        r_i,m:   i (= x, y or z) position of atom in pair m, m'
        ||r_m||: Euclidean distance between atoms in pair m, m'
        """

        derivative = np.zeros(shape=(len(atoms), 3))

        n_pairs = self.n_pairs
        euclidean_dists = [atoms.get_distance(i, j, mic=True)
                           for (i, j) in self.atom_pair_list]

        for m, (i, j) in enumerate(self.atom_pair_list):
            x_dist, y_dist, z_dist = [atoms[i].position[k] -
                                      atoms[j].position[k] for k in
                                      range(3)]

            x_i = x_dist / (n_pairs * euclidean_dists[m])
            y_i = y_dist / (n_pairs * euclidean_dists[m])
            z_i = z_dist / (n_pairs * euclidean_dists[m])

            derivative[i][:] = [x_i, y_i, z_i]
            derivative[j][:] = [-x_i, -y_i, -z_i]

        return derivative

    @property
    def n_pairs(self) -> int:
        """Number of pairs of atoms in this set"""
        return len(self.atom_pair_list)
