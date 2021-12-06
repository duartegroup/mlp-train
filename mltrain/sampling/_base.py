from abc import ABC, abstractmethod


class Function(ABC):
    """Function defining both a image and a gradient"""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Value of the function"""

    @abstractmethod
    def grad(self, *args, **kwargs):
        """Gradient of the function"""


class Constraint(ABC):
    """Abstract base class for an ASE constraint"""

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

