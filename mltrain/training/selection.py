import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
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

        self._configuration.single_point(method_name)
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


class MaxAtomicEnvDistance(SelectionMethod):

    def __init__(self,
                 threshold: float = 0.98):
        """
        Selection criteria based on the maximum distance between any of the
        training set and a new configuration. Evaluated based on the minimum
        SOAP kernel vector (K*) between a new configuration and prior training
        data

        -----------------------------------------------------------------------
        Arguments:
            threshold:
        """
        super().__init__()

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
            return

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

        return np.min(self._k_vec) < self.threshold

    @property
    def too_large(self) -> bool:
        return False

    @property
    def _n_training_envs(self) -> int:
        """Number of training environments available"""
        return len(self._k_vec)
