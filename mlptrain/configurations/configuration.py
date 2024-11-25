import mlptrain
import ase
import numpy as np
from typing import Optional, Union, List
from copy import deepcopy
from autode.atoms import AtomCollection, Atom
import autode.atoms
import ase.atoms
from mlptrain.log import logger
from mlptrain.energy import Energy
from mlptrain.forces import Forces
from mlptrain.box import Box
from mlptrain.configurations.calculate import run_autode


class Configuration(AtomCollection):
    """Configuration of atoms"""

    def __init__(
        self,
        atoms: Union[autode.atoms.Atoms, List[Atom], None] = None,
        charge: int = 0,
        mult: int = 1,
        box: Optional[Box] = None,
    ):
        """
        Set of atoms perhaps in a periodic box with an overall charge and
        spin multiplicity

        -----------------------------------------------------------------------
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

        # Collective variable values (obtained using PLUMED)
        self.plumed_coordinates: Optional[np.ndarray] = None

        self.time: Optional[float] = None  # Time in a trajectory
        self.n_ref_evals = 0  # Number of reference evaluations

    @property
    def ase_atoms(self) -> 'ase.atoms.Atoms':
        """
        ASE atoms for this configuration, absent of energy  and force
        properties.

        -----------------------------------------------------------------------
        Returns:
            (ase.atoms.Atoms): ASE atoms
        """
        _atoms = ase.atoms.Atoms(
            symbols=[atom.label for atom in self.atoms],
            positions=self.coordinates,
            pbc=self.box is not None,
        )

        if self.box is not None:
            _atoms.set_cell(cell=self.box.size)

        return _atoms

    def update_attr_from(self, configuration: 'Configuration') -> None:
        """
        Update system attributes from a configuration

        -----------------------------------------------------------------------
        Arguments:
            configuration:
        """

        self.charge = configuration.charge
        self.mult = configuration.mult
        self.box = deepcopy(configuration.box)

        return None

    def save_xyz(
        self,
        filename: str,
        append: bool = False,
        true: bool = False,
        predicted: bool = False,
    ) -> None:
        """
        Print this configuration as an extended xyz file where the first 4
        columns are the atom symbol, x, y, z and, if this configuration
        contains forces then add the x, y, z components of the force on as
        columns 4-7.

        -----------------------------------------------------------------------
        Arguments:
            filename:

            append: (bool) Append to the end of this xyz file?

            true: Save the true energy and forces

            predicted: Save the predicted energy and forces
        """
        # logger.info(f'Saving configuration to {filename}')

        a, b, c = [0.0, 0.0, 0.0] if self.box is None else self.box.size

        if true and predicted:
            raise ValueError(
                'Cannot save both predicted and true '
                f'quantities to {filename}'
            )

        if not (true or predicted):
            prop_str = ''

        else:
            energy = self.energy.predicted if predicted else self.energy.true
            prop_str = f'energy={energy if energy is not None else 0.:.8f} '

            prop_str += 'Properties=species:S:1:pos:R:3'
            forces = self.forces.predicted if predicted else self.forces.true
            if forces is not None:
                prop_str += ':forces:R:3'

        if not filename.endswith('.xyz'):
            logger.warning('Filename had no .xyz extension - adding')
            filename += '.xyz'

        with open(filename, 'a' if append else 'w') as exyz_file:
            print(
                f'{len(self.atoms)}\n'
                f'Lattice="{a:.6f} 0.000000 0.000000 '
                f'0.000000 {b:.6f} 0.000000 '
                f'0.000000 0.000000 {c:.6f}" '
                f'{prop_str}',
                file=exyz_file,
            )

            for i, atom in enumerate(self.atoms):
                x, y, z = atom.coord
                line = f'{atom.label} {x:.5f} {y:.5f} {z:.5f} '

                if (true or predicted) and forces is not None:
                    fx, fy, fz = forces[i]
                    line += f'{fx:.5f} {fy:.5f} {fz:.5f}'

                print(line, file=exyz_file)

        return None

    def single_point(
        self,
        method: Union[str, 'mlptrain.potentials._base.MLPotential'],
        n_cores: int = 1,
    ) -> None:
        """
        Run a single point energy and gradient (force) evaluation using
        either a reference method defined by a string (e.g. 'orca') or a
        machine learned potential (with a .predict) method.

        -----------------------------------------------------------------------
        Arguments:
            method:

            n_cores: Number of cores to use for the calculation
        """
        implemented_methods = ['xtb', 'orca', 'g09', 'g16']

        if isinstance(method, str) and method.lower() in implemented_methods:
            run_autode(self, method, n_cores=n_cores)
            self.n_ref_evals += 1
            return None

        elif hasattr(method, 'predict'):
            method.predict(self)

        else:
            raise ValueError(
                f'Cannot use {method} to predict energies and ' f'forces'
            )

        return None

    def __eq__(self, other) -> bool:
        """Another configuration is identical to this one"""
        eq = (
            isinstance(other, Configuration)
            and other.n_atoms == self.n_atoms
            and other.mult == self.mult
            and other.charge == self.charge
            and other.box == self.box
        )

        if eq and self.n_atoms > 0:
            rmsd = np.linalg.norm(self.coordinates - other.coordinates)
            return eq and rmsd < 1e-10
        return eq

    def copy(self) -> 'Configuration':
        return deepcopy(self)
