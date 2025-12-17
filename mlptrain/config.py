from autode.wrappers.keywords import GradientKeywords


class _ConfigClass:
    """
    MLP training configurations

    This class contains default parameters for electronic structure computations and training of available MLPs.
    Default settings for electronic structures is None to avoid accidentally running the wrong level of theory.
    The desired level can be specified by, e.g.
    ```
    from mlptrain.config import Config

    Config.orca_keywords = ['PBE', 'def2-SVP', 'EnGrad']
    Config.gaussian_keywords = ['PBEPBE', 'Def2SVP', 'Force(NoStep)', 'integral=ultrafinegrid']
    ```
    """

    n_cores = 4
    _orca_keywords = None
    _gaussian_keywords = None

    # Default parameters for a GAP potential
    gap_default_params = {
        'sigma_E': 10 ** (-4.0),  # eV
        'sigma_F': 10 ** (-2.0),
    }  # eV Å-1

    # Default SOAP parameters
    gap_default_soap_params = {
        'cutoff': 4.0,  # Å
        'n_sparse': 1000,
        'l_max': 6,  # n_max = 2 l_max
        'sigma_at': 0.5,  # Å
    }
    # ACE params
    ace_params = {
        'N': 4,  # maximum correlation order
        'r_cut': 4.0,  # outer cutoff of ACE
        'deg_pair': 5,  # Specify the pair potential
        'r_cut_pair': 5.0,
    }

    # NeQUIP params
    nequip_params = {'cutoff': 4.0, 'train_fraction': 0.9}

    # MACE params

    try:
        import torch

        mace_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        mace_device = 'cpu'

    mace_params = {
        'valid_fraction': 0.1,
        'max_num_epochs': 1200,
        'config_type_weights': '{"Default":1.0}',
        'model': 'MACE',
        'loss': 'weighted',
        'energy_weight': 1.0,
        'forces_weight': 5.0,
        'hidden_irreps': '128x0e + 128x1o',
        'batch_size': 10,
        'r_max': 5.0,
        'correlation': 3,
        'device': mace_device,
        'calc_device': 'cpu',
        'error_table': 'TotalMAE',
        'start_swa': None,
        'ema': True,
        'ema_decay': 0.99,
        'lr': 0.001,
        'patience': 50,
        'scheduler_patience': 20,
        'seed': 345,
        'amsgrad': True,
        'restart_latest': False,
        'save_cpu': True,
        'num_workers': 20,
        'max_L': 1,
        'dtype': 'float32',
        'cueq': False,
    }

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
        """Gaussian keywords must be gradient"""
        self._gaussian_keywords = value


# Singleton instance of the configuration
Config = _ConfigClass()
