from abc import ABC, abstractmethod
from typing import Optional


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

        if method_name is None:
            raise ValueError('Evaluating the absolute difference requires a '
                             'method name but None was present')

        if configuration.energy.predicted is None:
            configuration.single_point(mlp)

        configuration.single_point(method_name)
        return None

    @property
    def select(self) -> bool:
        """
        10 E_T > |E_predicted - E_true| > E_T
        """

        abs_dE = abs(self._configuration.energy.delta)
        return 10 * self.e_thresh > abs_dE > self.e_thresh

    @property
    def too_large(self) -> bool:
        """|E_predicted - E_true| > 10*E_T"""
        return abs(self._configuration.energy.delta) > 10 * self.e_thresh
