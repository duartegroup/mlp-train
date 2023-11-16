import mlptrain
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional
from mlptrain.descriptors import soap_kernel_vector
from mlptrain.log import logger


class SelectionMethod(ABC):
    """Active learning selection method

    NOTE: Should execute in serial
    """

    def __init__(self):
        """A selection method should determine whether its configuration
        should be selected during active learning"""

        self._configuration: Optional['mlptrain.Configuration'] = None

    @abstractmethod
    def __call__(
        self,
        configuration: 'mlptrain.Configuration',
        mlp: 'mlptrain.potentials.MLPotential',
        **kwargs,
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
    def __init__(self, e_thresh: float = 0.1):
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
            raise ValueError(
                'Evaluating the absolute difference requires a '
                'method name but None was present'
            )

        if configuration.energy.predicted is None:
            self._configuration.single_point(mlp)

        self._configuration.single_point(
            method_name, n_cores=kwargs.get('n_cores', 1)
        )
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
    def __init__(self, threshold: float = 0.999):
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

    def __call__(
        self,
        configuration: 'mlptrain.Configuration',
        mlp: 'mlptrain.potentials.MLPotential',
        **kwargs,
    ) -> None:
        """
        Evaluate the selection criteria

        -----------------------------------------------------------------------
        Arguments:
            configuration: Configuration to evaluate on

            mlp: Machine learning potential with some associated training data
        """
        if len(mlp.training_data) == 0:
            return None

        self._k_vec = soap_kernel_vector(
            configuration, configurations=mlp.training_data, zeta=8
        )
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

        return self.threshold**2 < np.max(self._k_vec) < self.threshold

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
