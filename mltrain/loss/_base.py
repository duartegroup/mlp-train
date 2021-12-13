from abc import ABC,abstractmethod
from typing import Optional


class LossValue(ABC, float):

    def __init__(self,
                 x,
                 error: Optional[float] = None):
        """
        Loss value with a possible associated error

        -----------------------------------------------------------------------
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
    def _err_str(self) -> str:
        """String containing the value and any associated error"""
        return '' if self.error is None else f'±{self.error}'


class LossFunction(ABC):

    def __init__(self,
                 method_name: Optional[str] = None):
        """
        Construct a loss function

        -----------------------------------------------------------------------
        Arguments:
            method_name: Name of the reference method to evaluate true
                         energies and forces
        """

        self.method_name = method_name

    @abstractmethod
    def __call__(self,
                 configurations: 'mltrain.ConfigurationSet',
                 mlp:            'mltrain.potentials._base.MLPotential',
                 **kwargs) -> LossValue:
        """Compute a loss value"""
