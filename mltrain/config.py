from autode.wrappers.keywords import *


class Config:
    """MLTrain configuration"""

    n_cores = 4
    _orca_keywords = GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])
    _gaussian_keywords = GradientKeywords(['PBEPBE', 'Def2SVP', 'Force(NoStep)', 'integral=ultrafinegrid'])

    # --------------------- Internal properties ---------------------------

    @property
    def orca_keywords(self):
        return self._orca_keywords

    @orca_keywords.setter
    def orca_keywords(self, value):
        """ORCA keywords must be gradient"""
        self._orca_keywords = GradientKeywords(value)

    @property
    def gaussian_keywords(self):
        return self._gaussian_keywords

    @gaussian_keywords.setter
    def gaussian_keywords(self, value):
        """ORCA keywords must be gradient"""
        self._gaussian_keywords = GradientKeywords(value)
