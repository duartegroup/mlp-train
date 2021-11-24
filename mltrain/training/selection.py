import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from scipy.stats import linregress
from mltrain.log import logger
from mltrain.descriptors import soap_kernel_vector


class SelectionMethod(ABC):
    """Active learning selection method

                    NOTE: Should execute in serial
    """

    def __init__(self):
        """A selection method should determine whether its configuration
        should be selected during active learning"""

        self._configuration: Optional['mltrain.Configuration'] = None

    @abstractmethod
    def __call__(self,
                 configuration: 'mltrain.Configuration',
                 mlp:           'mltrain.potentials.MLPotential',
                 **kwargs
                 ) -> None:
        """Evaluate the selector"""

    @property
    @abstractmethod
    def select(self) -> bool:
        """Should this configuration be selected?"""

    @property
    @abstractmethod
    def too_large(self) -> bool:
        """Is the error/discrepancy too large to be selected?"""

    @property
    @abstractmethod
    def n_backtrack(self) -> int:
        """
        Number of backtracking steps that this selection method should evaluate
        if the value is 'too_large'

        -----------------------------------------------------------------------
        Returns:
            (int):
        """

    def copy(self) -> 'SelectionMethod':
        return deepcopy(self)


class AbsDiffE(SelectionMethod):

    def __init__(self,
                 e_thresh: float = 0.1):
        """
        Selection method based on the absolute difference between the
        true and predicted total energies.

        -----------------------------------------------------------------------
        Arguments:
            e_thresh: E_T
        """
        super().__init__()

        self.e_thresh = e_thresh

    def __call__(self, configuration, mlp, **kwargs) -> None:
        """
        Evaluate the true and predicted energies, used to determine if this
        configuration should be selected.

        -----------------------------------------------------------------------
        Arguments:
            configuration: Configuration that may or may not be selected

            mlp: Machine learnt potential

            method_name: Name of the reference method to use
        """
        method_name = kwargs.get('method_name', None)
        self._configuration = configuration

        if method_name is None:
            raise ValueError('Evaluating the absolute difference requires a '
                             'method name but None was present')

        if configuration.energy.predicted is None:
            self._configuration.single_point(mlp)

        self._configuration.single_point(method_name,
                                         n_cores=kwargs.get('n_cores', 1))
        return None

    @property
    def select(self) -> bool:
        """
        10 E_T > |E_predicted - E_true| > E_T
        """

        abs_dE = abs(self._configuration.energy.delta)
        logger.info(f'|E_MLP - E_true| = {abs_dE:.4} eV')

        return 10 * self.e_thresh > abs_dE > self.e_thresh

    @property
    def too_large(self) -> bool:
        """|E_predicted - E_true| > 10*E_T"""
        return abs(self._configuration.energy.delta) > 10 * self.e_thresh

    @property
    def n_backtrack(self) -> int:
        return 10


class MaxAtomicEnvDistance(SelectionMethod):

    def __init__(self,
                 threshold: float = 0.999):
        """
        Selection criteria based on the maximum distance between any of the
        training set and a new configuration. Evaluated based on the similarity
        SOAP kernel vector (K*) between a new configuration and prior training
        data

        -----------------------------------------------------------------------
        Arguments:
            threshold: Value below which a configuration will be selected
        """
        super().__init__()

        if threshold < 0.1 or threshold >= 1.0:
            raise ValueError('Cannot have a threshold outside [0.1, 1]')

        self.threshold = float(threshold)
        self._k_vec = np.array([])

    def __call__(self,
                 configuration: 'mltrain.Configuration',
                 mlp:           'mltrain.potentials.MLPotential',
                 **kwargs) -> None:
        """
        Evaluate the selection criteria

        -----------------------------------------------------------------------
        Arguments:
            configuration: Configuration to evaluate on

            mlp: Machine learning potential with some associated training data
        """
        if len(mlp.training_data) == 0:
            logger.warning('Have no training data - unable to determine '
                           'criteria')
            return None

        self._k_vec = soap_kernel_vector(configuration,
                                         configurations=mlp.training_data)
        return None

    @property
    def select(self) -> bool:
        """
        Determine if this configuration should be selected, based on the
        minimum similarity between it and all of the training data

        -----------------------------------------------------------------------
        Returns:
            (bool): If this configuration should be selected
        """
        if self._n_training_envs == 0:
            return True

        _select = self.threshold**2 < np.max(self._k_vec) < self.threshold

        logger.info(f'max(K*) = {np.max(self._k_vec):.5}. Selecting: {_select}')
        return _select

    @property
    def too_large(self) -> bool:
        return np.max(self._k_vec) < self.threshold**2

    @property
    def n_backtrack(self) -> int:
        return 100

    @property
    def _n_training_envs(self) -> int:
        """Number of training environments available"""
        return len(self._k_vec)


class AccAbsDiffE(AbsDiffE):
    """
    Accelerated absolute energy difference method (|E_MLP - E_true| > E_T)
    by generating a regression model between SOAP vector similarity (cheap to
    calculate) and the absolute difference in total energies (slow)
    """

    def __init__(self,
                 e_thresh: float = 0.1,
                 min_r_sq: float = 0.9):
        """
        Accelerated absdiff method. Selection based on atomic environment
        similarity is turned on when R^2 reaches a threshold (min_r_sq)

        -----------------------------------------------------------------------
        Arguments:
            e_thresh: E_T

            min_r_sq: R^2 of the regression model below which |E_MLP - E_true|
                      is used as the selection criteria.
        """
        super().__init__(e_thresh=e_thresh)

        self._min_r_sq = min_r_sq
        self._regression_model = _RegressionModel()
        self._soap_selector = MaxAtomicEnvDistance()

    def __call__(self,
                 configuration: 'mltrain.Configuration',
                 mlp:           'mltrain.potentials.MLPotential',
                 **kwargs) -> None:
        """
        Evaluate the selection criteria

        -----------------------------------------------------------------------
        Arguments:
            configuration: Configuration to evaluate on

            mlp: Machine learning potential with some associated training data
        """
        if self._regression_model_is_good:
            self._soap_selector(configuration, mlp, **kwargs)

        else:
            super().__call__(configuration, mlp, **kwargs)
            self._regression_model.update(*self._x_y(mlp.training_data))

        return None

    @staticmethod
    def _x_y(configurations) -> Tuple[np.ndarray, np.ndarray]:
        """
        From a set of configurations extract the 'x' and 'y' values as the
        |E_MLP - E_true| and min(K*) respectively
        """
        x, y = [], []

        for idx, cfg in enumerate(configurations):

            if not cfg.energy.has_true_and_predicted:
                continue

            y.append(np.abs(cfg.energy.delta))
            x.append(np.max(soap_kernel_vector(cfg, configurations[:idx])))

        print(x)
        print(y)
        return np.array(x), np.array(y)

    @property
    def _regression_model_is_good(self) -> bool:
        return self._regression_model.r_sq > self._min_r_sq

    @property
    def select(self) -> bool:
        """Should this configuration be selected?"""
        if self._regression_model_is_good:
            return self._soap_selector.select

        else:
            return super().select

    @property
    def too_large(self) -> bool:
        """Is the error too large for this configuration to be selected"""
        if self._regression_model_is_good:
            return self._soap_selector.too_large

        else:
            return super().too_large


class _RegressionModel:

    def __init__(self):
        """
        Regression model built from some x and y data, making use of
        scipy.stats.linregres
        """
        self._linrg_result = None

    def update(self,
               x:    np.ndarray,
               y:    np.ndarray,
               plot: bool = False) -> None:
        """
        Update the regression model

        -----------------------------------------------------------------------
        Arguments:
            x:
            y:
            plot: Whether to plot the true and fitted models
        """
        if len(x) == 0:
            logger.warning('Cannot update the regression model with no data')
            return None

        logger.info(f'Updating regression model with {len(x)} points')

        self._linrg_result = linregress(x, y)

        if plot:
            plt.scatter(x, y)

            m, c = self._linrg_result.slope, self._linrg_result.intercept

            plt.plot([min(x), max(x)],
                     [m*min(x) + c, m*max(x) + c])

            self._set_plot_params()
            plt.savefig(f'reg_{id(self)}')

        return None

    @property
    def r_sq(self) -> float:
        """R^2"""

        if self._linrg_result is None:
            logger.warning('No regression performed - returning R^2 = 0.0')
            return 0.0

        return self._linrg_result.rvalue ** 2

    def _set_plot_params(self) -> None:

        plt.xlabel('max(K*)')
        plt.ylabel('|E_MLP - E_true|')
        plt.tight_layout()
