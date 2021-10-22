import numpy as np
from typing import Optional


class Forces:
    """Forces in units of eV / Å"""

    @property
    def delta(self) -> np.ndarray:
        """
        Difference between true and predicted forces

        Returns:
            (np.ndarray):  F_true - F_predicted. Shape = (n_atoms, 3)

        Raises:
            (ValueError): If at least one set of forces is not defined
        """

        if self.true is None:
            raise ValueError('Cannot calculate ∆F. No true forces')

        if self.predicted is None:
            raise ValueError('Cannot calculate ∆F. No predicted forces')

        if self.true.shape != self.predicted.shape:
            raise ValueError('Cannot calculate ∆F. Shape mismatch')

        return self.true - self.predicted

    def __init__(self,
                 predicted: Optional[np.ndarray] = None,
                 true:      Optional[np.ndarray] = None):
        """

        Arguments:
            predicted:
            true:
        """

        self.predicted = predicted
        self.true = true
