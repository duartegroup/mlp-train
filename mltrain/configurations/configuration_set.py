from typing import Optional, List


class ConfigurationSet(list):
    """A set of configurations"""

    @property
    def true_energies(self) -> List[float]:
        """True calculated energies"""
        return [c.energy.true for c in self]

    @property
    def predicted_energies(self) -> List[float]:
        """Predicted energies using a MLP"""
        return [c.energy.predicted for c in self]

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
