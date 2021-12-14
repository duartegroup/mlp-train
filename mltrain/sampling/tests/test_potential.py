import numpy as np
from mltrain.potentials._base import MLPotential
from ase.calculators.calculator import Calculator


class HarmonicPotential(Calculator):

    __test__ = False

    def get_potential_energy(self, atoms):

        r = atoms.get_distance(0, 1)

        return (r - 1)**2

    def get_forces(self, atoms):

        derivative = np.zeros((len(atoms), 3))

        r = atoms.get_distance(0, 1)

        x_dist, y_dist, z_dist = [atoms[0].position[j] - atoms[1].position[j]
                                  for j in range(3)]

        x_i, y_i, z_i = (x_dist / r), (y_dist / r), (z_dist / r)

        derivative[0][:] = [x_i, y_i, z_i]
        derivative[1][:] = [-x_i, -y_i, -z_i]

        force = -2 * derivative * (r - 1)

        return force


class TestPotential(MLPotential):

    __test__ = False

    def __init__(self,
                 name: str,
                 system=None):

        super().__init__(name=name, system=system)

    @property
    def ase_calculator(self):

        return HarmonicPotential()

    def _train(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""

    def requires_atomic_energies(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""

    def requires_non_zero_box_size(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""
