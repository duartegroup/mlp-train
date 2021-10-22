from typing import Optional
from autode.atoms import AtomCollection, Atoms
from ase.atoms import Atoms
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

    @property
    def ase_atoms(self) -> 'ase.atoms.Atoms':
        """
        ASE atoms for this configuration, absent of energy  and force
        properties

        Returns:
            (ase.atoms.Atoms): ASE atoms
        """

        return Atoms(symbols=[atom.label for atom in self.atoms],
                     positions=self.coordinates)
