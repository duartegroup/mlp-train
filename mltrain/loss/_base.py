from abc import ABC,abstractmethod
from typing import Optional


class LossValue(ABC, float):

    def __init__(self,
                 x,
                 error: Optional[float] = None):
        """
        Loss value with a possible associated error

        Arguments:
            x (float | int): Value

        Keyword Arguments:
            error (float | None):
        """

        float.__init__(float(x))
        self.error: Optional[float] = error

    @abstractmethod
    def __repr__(self) -> str:
        """Representation of this loss"""

    @property
    def _value_str(self) -> str:
        """String containing the value and any associated error"""
        return f'{self}' if self.error is None else f'{self}±{self.error}'


class LossFunction(ABC):

    @abstractmethod
    def __call__(self,
                 configurations: 'mltrain.ConfigurationSet',
                 mlp:            'mltrain.potentials._base.MLPotential'
                 ) -> LossValue:
        """Compute a loss value"""
