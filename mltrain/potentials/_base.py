import mltrain as mlt
from mltrain.training.active import train as al_train
from abc import ABC, abstractmethod
from typing import Optional


class MLPotential(ABC):

    def __init__(self,
                 name:   str,
                 system: 'mltrain.System'):
        """
        Machine learnt potential. Name defines the name of the potential
        which will be saved. Training data is populated

        -----------------------------------------------------------------------
        Arguments:
            name: Name of the potential
            system: System for which this potential is defined
        """
        self.name = str(name)
        self.system = system

        self._training_data = mlt.ConfigurationSet()

    def train(self,
              configurations: Optional['mltrain.ConfigurationSet'] = None):
        """
        Train this potential on a set of configurations

        -----------------------------------------------------------------------
        Arguments:
            configurations: Set of configurations to train on, if None then
                            will use self._training_data
        """
        if configurations is not None:
            self._training_data = configurations

        if len(self.training_data) == 0:
            raise RuntimeError(f'Failed to train {self.__name__}({self.name}) '
                               f'had no training configurations')

        self._train()
        return None

    @abstractmethod
    def _train(self) -> None:
        """Train this potential on self._training_data"""

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

    @property
    def training_data(self) -> 'mltrain.ConfigurationSet':
        """Training data which this potential was trained on

        Returns:
            (mltrain.ConfigurationSet):
        """
        return self._training_data

    @training_data.setter
    def training_data(self, value: Optional['mltrain.ConfigurationSet']):
        """Set the training date for this MLP"""

        if value is None:
            self._training_data.clear()

        elif isinstance(value, mlt.ConfigurationSet):
            self._training_data = value

        else:
            raise ValueError(f'Cannot set the training data for {self.name} '
                             f'with {value}')

    def al_train(self,
                 method_name: str,
                 **kwargs) -> None:
        """
        Train this MLP using active learning (AL) using a defined reference
        method

        Arguments:
            method_name:
            **kwargs:  Keyword arguments passed to mlt.training.active.train()
        """
        return al_train(self, method_name=method_name, **kwargs)
