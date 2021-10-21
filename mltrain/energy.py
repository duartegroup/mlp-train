from typing import Optional


class Energy:
    """Energy in units of eV"""

    def __init__(self,
                 predicted: Optional[float] = None,
                 true:      Optional[float] = None):
        """

        Arguments:
            predicted:
            true:
        """

        self.predicted = predicted
        self.true = true
