from autode.wrappers.keywords import *


class _ConfigClass:
    """mlptrain configuration"""

    n_cores = 4
    _orca_keywords = ['PBE', 'def2-SVP', 'EnGrad']
    _gaussian_keywords = ['PBEPBE', 'Def2SVP', 'Force(NoStep)', 'integral=ultrafinegrid']

    # Default parameters for a GAP potential
    gap_default_params = {'sigma_E': 10**(-4.0),        # eV
                          'sigma_F': 10**(-2.0)}        # eV Å-1

    # Default SOAP parameters
    gap_default_soap_params = {'cutoff':   4.0,         # Å
                               'n_sparse': 1000,
                               'l_max':    6,           # n_max = 2 l_max
                               'sigma_at': 0.5          # Å
                               }

    # NeQUIP params
    nequip_params = {'cutoff': 4.0,
                     'train_fraction': 0.9}

    # --------------------- Internal properties ---------------------------

    @property
    def orca_keywords(self):
        return GradientKeywords(self._orca_keywords)

    @orca_keywords.setter
    def orca_keywords(self, value):
        """ORCA keywords must be gradient"""
        self._orca_keywords = value

    @property
    def gaussian_keywords(self):
        return GradientKeywords(self._gaussian_keywords)

    @gaussian_keywords.setter
    def gaussian_keywords(self, value):
        """ORCA keywords must be gradient"""
        self._gaussian_keywords = value


# Singleton instance of the configuration
Config = _ConfigClass()
