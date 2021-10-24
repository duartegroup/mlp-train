import numpy as np
from typing import Optional, List, Union
from mltrain.log import logger


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

    def save(self,
             filename:  str,
             true:      bool = False,
             predicted: bool = False
             ) -> None:
        """Save these configurations to a file

        Arguments:
            filename:

            true: Save 'true' energies and forces, if they exist

            predicted: Save the MLP predicted energies and forces, if they
                       exist.
        """

        if len(self) == 0:
            logger.error(f'Failed to save {filename}. Had no configurations')
            return

        if self[0].energy.true is not None and not (predicted or true):
            logger.warning('Save called without defining what energy and '
                           'forces to print. Had true energies to using those')
            true = True

        for configuration in self:
            configuration.save(filename,
                               true=true,
                               predicted=predicted,
                               append=True)
        return None
