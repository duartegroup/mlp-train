class Energy:
    """Energy in units of eV"""

    def __init__(
        self,
        predicted: float | None = None,
        true: float | None = None,
        bias: float | None = None,
        inherited_bias: float | None = None,
    ):
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
        self.inherited_bias = inherited_bias

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
