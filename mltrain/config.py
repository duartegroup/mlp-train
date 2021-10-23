from autode.wrappers.keywords import *


class Config:
    """MLTrain configuration"""

    n_cores = 4
    _orca_keywords = GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])

    # --------------------- Internal properties ---------------------------

    @property
    def orca_keywords(self):
        return self._orca_keywords

    @orca_keywords.setter
    def orca_keywords(self, value):
        """ORCA keywords must be gradient"""
        self._orca_keywords = GradientKeywords(value)
