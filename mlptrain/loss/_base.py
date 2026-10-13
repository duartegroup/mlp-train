from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mlptrain

if TYPE_CHECKING:
    from mlptrain.potentials import MLPotential


class LossValue(ABC, float):
    def __init__(self, x, error: float | None = None):
        """
        Loss value with a possible associated error

        -----------------------------------------------------------------------
        Arguments:
            x (float | int): Value

        Keyword Arguments:
            error (float | None):
        """

        float.__init__(float(x))
        self.error: float | None = error

    @abstractmethod
    def __repr__(self) -> str:
        """Representation of this loss"""

    @property
    def _err_str(self) -> str:
        """String containing the value and any associated error"""
        return '' if self.error is None else f'±{self.error}'


class LossFunction(ABC):
    def __init__(self, method_name: str | None = None):
        """
        Construct a loss function

        -----------------------------------------------------------------------
        Arguments:
            method_name: Name of the reference method to evaluate true
                         energies and forces
        """

        self.method_name = method_name

    @abstractmethod
    def __call__(
        self,
        configurations: mlptrain.ConfigurationSet,
        mlp: MLPotential,
        **kwargs,
    ) -> LossValue:
        """Compute a loss value"""
