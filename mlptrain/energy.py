from typing import Optional


class Energy:
    """Energy in units of eV"""

    def __init__(self,
                 predicted: Optional[float] = None,
                 true:      Optional[float] = None,
                 bias:      Optional[float] = None):
        """
        Energy

        -----------------------------------------------------------------------
        Arguments:
            predicted:
            true:
            bias:
        """

        self.predicted = predicted
        self.true = true
        self.bias = bias

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
