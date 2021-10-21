import numpy as np
from typing import Optional


class Forces:
    """Forces in units of eV / Å"""

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
