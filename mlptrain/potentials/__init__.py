# Useful for typing
from ._base import MLPotential

from .gap.gap import GAP
from .ace.ace import ACE
from .nequip._nequip import NequIP

__all__ = ['GAP', 'ACE', 'NequIP', 'MACE', 'MLPotential']


# Lazy load MACE to improve import time
# TODO: We could probably load the others lazily as well
def __getattr__(name):
    if name == 'MACE':
        from .mace.mace import MACE

        return MACE
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
