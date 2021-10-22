import mltrain as mlt
from abc import ABC, abstractmethod


class MLPotential(ABC):

    @abstractmethod
    def train(self,
              configurations: 'mltrain.ConfigurationSet',
              **kwargs):
        """Train this potential on a set of configurations"""

    @property
    @abstractmethod
    def ase_calculator(self) -> 'ase.calculators.calculator.Calculator':
        """Generate an ASE calculator for this potential"""

    def predict(self, *args) -> None:
        """
        Predict energies and forces using a MLP in serial

        Arguments:
            args (mltrain.ConfigurationSet | mltrain.Configuration):
        """
        all_configurations = mlt.ConfigurationSet()

        for arg in args:
            if isinstance(arg, mlt.ConfigurationSet):
                all_configurations += arg

            elif isinstance(arg, mlt.Configuration):
                all_configurations.append(arg)

            else:
                raise ValueError('Cannot predict the energy and forces on '
                                 f'{type(arg)}')

        for configuration in all_configurations:
            atoms = configuration.ase_atoms
            atoms.set_calculator(self.ase_calculator)

            # Evaluate predicted energies and forces
            configuration.energy.predicted = atoms.get_potential_energy()
            configuration.forces.predicted = atoms.get_forces()

        return None
