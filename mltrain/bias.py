from abc import ABC, abstractmethod
import numpy as np


class Bias:
    """Modifies the forces and energy of a set of atoms"""

    def __init__(self, coordinate, reference, kappa, function):
        """
        :param coordinate: (list) Indices of the atoms which define the
                           reaction coordinate. E.g., [[0, 1], [2, 3]]
        :param reference: (float) Value of the reference value, ξ_i,
                          used in umbrella sampling
        :param kappa: (float) Value of the spring_const, κ,
                      used in umbrella sampling
        :param function: (str) Type of bias function to use. 'Euclidean' is the
                         only implemented function currently
        """

        self.coordinate = coordinate
        self.ref = reference
        self.kappa = kappa
        self.function = function

        assert function == 'Euclidean'
        func = EuclideanDistance()

        self.bias_function = BiasPotential(func, self.ref, self.kappa)

    def adjust_forces(self, atoms, forces):

        bias = -self.bias_function.grad(self.coordinate, atoms)

        forces[:] = forces[:] + bias

        return None

    def adjust_potential_energy(self, atoms):

        bias = self.bias_function(self.coordinate, atoms)

        return bias

    def adjust_positions(self, atoms, newpositions):
        """Method required for ASE but not needed in ml-train"""
        return None


class Function(ABC):

    @abstractmethod
    def __call__(self, r, a):
        """Value of the potential for set of r coordinates in atoms, a"""

    @abstractmethod
    def grad(self, r, a):
        """
        Gradient of the potential dU/dr for set of r coordinates in
        atoms, a
        """


class BiasPotential(Function):
    """Harmonic biasing potential"""

    def __init__(self,
                 func:  Function,
                 ref: float,
                 kappa: float):
        """
        Harmonic biasing potential: ω = κ/2 (f(r) - s)^2
        -----------------------------------------------------------------------
        Arguments:
            func: Function to transform the coordinate

            ref: Reference value for the bias

            kappa: Strength of the biasing potential
        """

        self.func = func
        self.ref = ref
        self.kappa = kappa

    def __call__(self, r, a):
        """Value of the bias for set of r coordinates in atoms, a"""
        return 0.5 * self.kappa * (self.func(r, a) - self.ref)**2

    def grad(self, r, a):
        """
        Gradient of the biasing potential for set of r coordinates in
        atoms, a
        """
        return self.kappa * self.func.grad(r, a) * (self.func(r, a) - self.ref)


class EuclideanDistance(Function):
    """Euclidean distance between each pair of atoms specified"""

    def __call__(self, r, atoms):
        """Average distance between atom pairs in r coordinates"""
        num_pairs = len(r)

        euclidean_dists = [atoms.get_distance(r[i][0], r[i][1], mic=True)
                           for i in range(num_pairs)]

        return np.mean(euclidean_dists)

    def grad(self, r, atoms):
        """r is the coordinate indexes of pairs. E.g., [[1,0], [2,4]]"""

        derivitive_vector = np.zeros((len(atoms), 3))

        num_pairs = len(r)
        euclidean_dists = [atoms.get_distance(r[i][0], r[i][1], mic=True)
                           for i in range(num_pairs)]

        for m, pair in enumerate(r):
            x_dist, y_dist, z_dist = [atoms[pair[0]].position[j] -
                                      atoms[pair[1]].position[j] for j in
                                      range(3)]

            x_i = x_dist / (num_pairs * euclidean_dists[m])
            y_i = y_dist / (num_pairs * euclidean_dists[m])
            z_i = z_dist / (num_pairs * euclidean_dists[m])

            derivitive_vector[pair[0]][:] = [x_i, y_i, z_i]
            derivitive_vector[pair[1]][:] = [-x_i, -y_i, -z_i]

        return derivitive_vector
