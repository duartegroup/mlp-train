from typing import Optional


class Energy:
    """Energy in units of eV"""

    def __init__(self,
                 predicted: Optional[float] = None,
                 true:      Optional[float] = None):
        """
        Energy

        -----------------------------------------------------------------------
        Arguments:
            predicted:
            true:
        """

        self.predicted = predicted
        self.true = true

    @property
    def delta(self) -> float:
        """
        Difference between true and predicted energies

        -----------------------------------------------------------------------
        Returns:
            (float):  E_true - E_predicted

        Raises:
            (ValueError): If at least one energy is not defined
        """

        if self.true is None:
            raise ValueError('Cannot calculate ∆E. No true energy')

        if self.predicted is None:
            raise ValueError('Cannot calculate ∆E. No predicted energy')

        return self.true - self.predicted

    @property
    def has_true_and_predicted(self) -> bool:
        """
        Does this energy have both true and predicted values

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        return self.true is not None and self.predicted is not None
