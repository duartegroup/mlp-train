from typing import Optional
from autode.atoms import AtomCollection, Atoms
from mltrain.energy import Energy
from mltrain.forces import Forces


class Configuration(AtomCollection):
    """Configuration of atoms"""

    def __init__(self,
                 atoms:  Optional[Atoms] = None,
                 charge: int = 0,
                 mult:   int = 0):
        """

        Arguments:
            atoms:
            charge:
            mult:
        """

        super().__init__(atoms=atoms)

        self.charge = charge
        self.mult = mult

        self.energy = Energy()
        self.forces = Forces()

