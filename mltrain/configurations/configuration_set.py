import numpy as np
from typing import Optional, List, Union


class ConfigurationSet(list):
    """A set of configurations"""

    @property
    def true_energies(self) -> List[Optional[float]]:
        """True calculated energies"""
        return [c.energy.true for c in self]

    @property
    def predicted_energies(self) -> List[Optional[float]]:
        """Predicted energies using a MLP"""
        return [c.energy.predicted for c in self]

    @property
    def lowest_energy(self) -> 'mltrain.Configuration':
        """
        Determine the lowest energy configuration in this set based on the
        true energies. If not evaluated then returns the first configuration

        Returns:
            (mltrain.Configuration):
        """
        if len(self) == 0:
            raise ValueError('No lowest energy configuration in an empty set')

        energies = [e if e is not None else np.inf for e in self.true_energies]
        return self[np.argmin(energies)]

    def append(self,
               value: Optional['mltrain.Configuration']) -> None:
        """
        Append an item onto these set of configurations. None will not be
        appended

        Arguments:
            value:
        """

        if value is None:
            return

        return super().append(value)

    def compare(self,
                *args: Union['mltrain.potentials.MLPotential', str]) -> None:
        """
        Compare methods e.g. a MLP to a ground truth reference method over
        these set of configurations

        Args:
            *args: Strings defining the method or MLPs
        """
        raise NotImplementedError

    def save(self) -> None:

        raise NotImplementedError
