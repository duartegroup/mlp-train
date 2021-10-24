from typing import Optional, Union
from copy import  deepcopy
from autode.atoms import AtomCollection, Atoms
from ase.atoms import Atoms
from mltrain.log import logger
from mltrain.energy import Energy
from mltrain.forces import Forces
from mltrain.box import Box
from mltrain.configurations.calculate import run_autode


class Configuration(AtomCollection):
    """Configuration of atoms"""

    def __init__(self,
                 atoms:  Optional[Atoms] = None,
                 charge: int = 0,
                 mult:   int = 0,
                 box:    Optional[Box] = None):
        """

        Arguments:
            atoms:
            charge:
            mult:
            box: Optional box, if None then
        """
        super().__init__(atoms=atoms)

        self.charge = charge
        self.mult = mult
        self.box = box

        self.energy = Energy()
        self.forces = Forces()

        self.time = 0         # Time in a trajectory of this configuration
        self.n_ref_evals = 0  # Number of reference evaluations on this config.

    @property
    def ase_atoms(self) -> 'ase.atoms.Atoms':
        """
        ASE atoms for this configuration, absent of energy  and force
        properties.

        Returns:
            (ase.atoms.Atoms): ASE atoms
        """
        _atoms = Atoms(symbols=[atom.label for atom in self.atoms],
                       positions=self.coordinates,
                       pbc=self.box is not None)

        if self.box is not None:
            _atoms.set_cell(cell=self.box.size)

        return _atoms

    def update_attr_from(self,
                         configuration: 'Configuration') -> None:
        """
        Update system attributes from a configuration

        Arguments:
            configuration:
        """

        self.charge = configuration.charge
        self.mult = configuration.mult
        self.box = deepcopy(configuration.box)

        return None

    def save(self,
             filename:  str,
             append:    bool = False,
             true:      bool = False,
             predicted: bool = False
             ) -> None:
        """
        Print this configuration as an extended xyz file where the first 4
        columns are the atom symbol, x, y, z and, if this configuration
        contains forces then add the x, y, z components of the force on as
        columns 4-7.

        -----------------------------------------------------------------------
        Arguments:
            filename:

        Keyword Arguments:
            append: (bool) Append to the end of this exyz file?
        """
        logger.info(f'Saving configuration as {filename}')

        a, b, c = [0., 0., 0.] if self.box is None else self.box.size

        if true and predicted:
            raise ValueError('Cannot save both predicted and true '
                             f'quantities to {filename}')

        if not (true or predicted):
            prop_str = ''

        else:
            energy = self.energy.predicted if predicted else self.energy.true
            prop_str = f'energy={energy if energy is not None else 0.:.8f}'

            prop_str += 'Properties=species:S:1:pos:R:3'
            forces = self.forces.predicted if predicted else self.forces.true
            if forces is not None:
                prop_str += ':forces:R:3'

        if not filename.endswith('.xyz'):
            logger.warning('Filename had no .xyz extension - adding')
            filename += '.xyz'

        with open(filename, 'a' if append else 'w') as exyz_file:
            print(f'{len(self.atoms)}\n'
                  f'Lattice="{a:.6f} 0.000000 0.000000 '
                  f'0.000000 {b:.6f} 0.000000 '
                  f'0.000000 0.000000 {c:.6f}" '
                  f'{prop_str}',
                  file=exyz_file)

            for i, atom in enumerate(self.atoms):
                x, y, z = atom.coord
                line = f'{atom.label} {x:.5f} {y:.5f} {z:.5f} '

                if (true or predicted) and forces is not None:
                    fx, fy, fz = forces[i]
                    line += f'{fx:.5f} {fy:.5f} {fz:.5f}'

                print(line, file=exyz_file)

        return None

    def single_point(self,
                     method:  Union[str, 'mltrain.potentials.MLPotential'],
                     n_cores: int = 1
                     ) -> None:
        """
        Run a single point energy and gradient (force) evaluation using
        either a reference method defined by a string (e.g. 'orca') or a
        machine learned potential (with a .predict) method.

        Arguments:
            method:
        """
        implemented_methods = ['xtb', 'orca', 'g09', 'g16']

        if isinstance(method, str) and method.lower() in implemented_methods:
            run_autode(self, method, n_cores=n_cores)
            self.n_ref_evals += 1
            return None

        elif hasattr(method, 'predict'):
            method.predict(self)

        else:
            raise ValueError(f'Cannot use {method} to predict energies and '
                             f'forces')

        return None
