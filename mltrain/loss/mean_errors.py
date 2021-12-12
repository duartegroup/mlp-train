import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import bootstrap
from mltrain.loss._base import LossValue, LossFunction


class _DeltaLossFunction(LossFunction, ABC):
    """Error measure that depends on E_true - E_predicted"""

    loss_type = None

    def __call__(self,
                 configurations: 'mltrain.ConfigurationSet',
                 mlp:            'mltrain.potentials.MLPotential',
                 **kwargs) -> LossValue:
        """Calculate the value of the loss

        -----------------------------------------------------------------------
        Arguments:
            configurations: Set of configurations to evaluate over

            mlp: Potential to use
        """

        if self.loss_type is None:
            raise NotImplementedError(f'{self} did not define loss_type')

        if 'method_name' in kwargs:
            self.method_name = kwargs.pop('method_name')

        if len(kwargs) > 0:
            raise ValueError(f'Unknown keyword arguments: {kwargs}')

        delta_Es = self._delta_energies(configurations, mlp)
        std_error = bootstrap(delta_Es, self.statistic).standard_error

        return self.loss_type(self.statistic(delta_Es), error=std_error)

    def _delta_energies(self, cfgs, mlp):
        """Evaluate E_true - E_predicted along a set of configurations"""

        for idx, configuration in enumerate(cfgs):

            if configuration.energy.true:
                if self.method_name is not None:
                    configuration.single_point(method=self.method_name)

                else:
                    raise RuntimeError(f'Cannot compute loss for configuration '
                                       f'{idx}- a true energies was not present')

            if configuration.energy.predicted is None:
                mlp.predict(configuration)

        return np.array([c.energy.delta for c in cfgs])

    @staticmethod
    @abstractmethod
    def statistic(arr: np.ndarray) -> float:
        """Error measure over an array of values"""


class RMSEValue(LossValue):

    def __repr__(self):
        return f'RMSE({self._value_str})'


class RMSE(_DeltaLossFunction):
    """ RMSE = √(1/N Σ_i (y_i^predicted - y_i^true)^2)"""

    loss_type = RMSEValue

    @staticmethod
    def statistic(arr: np.ndarray) -> float:
        return np.sqrt(np.mean(np.square(arr)))


class MADValue(LossValue):

    def __repr__(self):
        return f'MAD({self._value_str})'


class MAD(LossFunction):
    """ MAD = 1/N √(Σ_i |y_i^predicted - y_i^true|)"""

    loss_type = MADValue

    @staticmethod
    def statistic(arr: np.ndarray) -> float:
        return float(np.mean(np.abs(arr)))
