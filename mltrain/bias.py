from mltrain.log import logger
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class Constraint:

    @abstractmethod
    def adjust_forces(self, atoms, forces):
        """Adjust the forces of a set of atoms using a the gradient of the bias
        function"""

    @abstractmethod
    def adjust_potential_energy(self, atoms):
        """Adjust the energy of a set of atoms using the bias function"""

    @abstractmethod
    def adjust_positions(self, atoms, newpositions):
        """Method required for ASE but not used in ml-train"""


class Function(ABC):

    @abstractmethod
    def __call__(self, atom_pair_list, a):
        """Value of the potential for atom_pair_list in atoms, a"""

    @abstractmethod
    def grad(self, atom_pair_list, a):
        """
        Gradient of the potential dU/dr for atom_pair_list in atoms, a
        """


class Bias(Constraint, Function):
    """Modifies the forces and energy of a set of ASE atoms under a bias"""

    def __init__(self,
                 kappa: float,
                 reference: Optional[float] = None,
                 **kwargs):
        """
        Bias that modifies the forces and energy of a set of atoms under a
        harmonic bias function.

        Harmonic biasing potential: ω = κ/2 (f(r) - s)^2

        e.g. bias = mlt.bias.Bias(to_average=[[0, 1]], reference=2, kappa=10)

        -----------------------------------------------------------------------
        Arguments:

            kappa: Value of the spring_const, κ, used in umbrella sampling

            reference: Value of the reference value, ξ_i, used in umbrella
                       sampling

        -------------------
        Keyword Arguments:

            {to_add, to_subtract, to_average}: (list) Indicies of the atoms
            which are combined in some way to define the reaction rxn_coord
        """
        self.s = reference
        self.kappa = kappa

        try:
            assert len(kwargs) == 1

        except AssertionError:
            logger.error("Must specify one rxn_coord combination method")

        if 'to_add' in kwargs:
            self.rxn_coord = kwargs['to_add']
            raise NotImplementedError("Addition reaction rxn_coord not yet "
                                      "implemented")

        elif 'to_subtract' in kwargs:
            self.rxn_coord = kwargs['to_subtract']
            raise NotImplementedError("Subtract reaction rxn_coord not yet "
                                      "implemented")

        elif 'to_average' in kwargs:
            self.rxn_coord = kwargs['to_average']
            self.func = AverageDistance()

        else:
            raise ValueError("Bias must specify one of to_add, to_subtract "
                             "or to_average!")

    def __call__(self, atom_pair_list, atoms):
        """Value of the bias for set of atom pairs in atoms"""

        return 0.5 * self.kappa * (self.func(atom_pair_list, atoms) - self.s)**2

    def grad(self, atom_pair_list, atoms):
        """Gradient of the biasing potential a set of atom pairs in atoms"""

        return self.kappa * self.func.grad(
            atom_pair_list, atoms) * (self.func(atom_pair_list, atoms) - self.s)

    def adjust_potential_energy(self, atoms):
        """Adjust the energy of a set of atoms using the bias function"""

        bias = self.__call__(self.rxn_coord, atoms)

        return bias

    def adjust_forces(self, atoms, forces):
        """Adjust the forces of a set of atoms using a the gradient of the bias
         function"""

        bias = -self.grad(self.rxn_coord, atoms)

        forces[:] = forces[:] + bias

        return None

    def adjust_positions(self, atoms, newpositions):
        """Method required for ASE but not used in ml-train"""
        return None


class AverageDistance(Function):
    """Average Euclidean distance between each pair of atoms specified"""

    def __call__(self, atom_pair_list, atoms):
        """Average distance between atom pairs"""

        euclidean_dists = [atoms.get_distance(i, j, mic=True)
                           for (i, j) in atom_pair_list]

        return np.mean(euclidean_dists)

    def grad(self, atom_pair_list, atoms):
        """Gradient of the average distance between atom pairs"""

        derivitive_vector = np.zeros((len(atoms), 3))

        num_pairs = len(atom_pair_list)
        euclidean_dists = [atoms.get_distance(i, j, mic=True)
                           for (i, j) in atom_pair_list]

        for m, (i, j) in enumerate(atom_pair_list):
            x_dist, y_dist, z_dist = [atoms[i].position[k] -
                                      atoms[j].position[k] for k in
                                      range(3)]

            x_i = x_dist / (num_pairs * euclidean_dists[m])
            y_i = y_dist / (num_pairs * euclidean_dists[m])
            z_i = z_dist / (num_pairs * euclidean_dists[m])

            derivitive_vector[i][:] = [x_i, y_i, z_i]
            derivitive_vector[j][:] = [-x_i, -y_i, -z_i]

        return derivitive_vector
