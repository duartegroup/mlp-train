from abc import ABC, abstractmethod


class MLPotential(ABC):

    @abstractmethod
    def train(self,
              configurations: 'mltrain.ConfigurationSet'):
        """Train this potential on a set of configurations"""

    @abstractmethod
    def ase_calculator(self) -> 'ase.calculators.calculator.Calculator':
        """Generate an ASE calculator for this potential"""
