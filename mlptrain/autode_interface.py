"""
Helper class for interfacing mlptrain with autoDE.
"""
import os
import numpy as np
import mlptrain as mlt
from mlptrain.sampling.md import _convert_ase_traj
from mlptrain.box import Box
from autode.wrappers.methods import ExternalMethodOEG
from autode.wrappers.keywords import KeywordsSet
from autode.calculations.calculation import Calculation
from autode.values import PotentialEnergy
from autode.utils import work_in_tmp_dir as work_in_tmp_dir_ade
from copy import deepcopy
from mlptrain.log import logger
from ase.constraints import Hookean

ev_to_ha = 1.0 / 27.2114

def from_autode_to_ase(molecule, cell_size=100):
    """
    Convert autode.molecule to ase.atoms and maintain the constraint
    generated during ade.pes.RelaxedPESnD calculation.
    """
    from ase.atoms import Atoms

    atoms = Atoms(
        symbols=[atom.label for atom in molecule.atoms],
        positions=molecule.coordinates,
        pbc=True,
    )

    atoms.set_cell([(cell_size, 0, 0), (0, cell_size, 0), (0, 0, cell_size)])

    c = []
    for (i, j), dist in molecule.constraints.distance.items():
        c.append(Hookean(a1=i, a2=j, k=50, rt=dist))
    atoms.set_constraint(c)
    return atoms


class MLPEST(ExternalMethodOEG):
    """
    Custom class of machine learning potential fitted for autode package.
    Original code provided by T. Young.

    Arguments:
        mlp
    """

    def __init__(
        self,
        mlp: mlt.potentials.MLPotential,
        action: list[str],
        path: str,
        opt_fmax: float = 0.01,
        kept_file_exts: tuple[str] = ('.xyz'),
    ):
        super().__init__(
            executable_name='mlp',
            keywords_set=KeywordsSet(),
            path='',
            doi_list=[],
            implicit_solvation_type=None,
        )

        self.path = path
        self.mlp = mlp
        self.action = deepcopy(action)
        self.opt_fmax = opt_fmax
        self.kept_file_exts = kept_file_exts

    @property
    def is_available(self):
        return True

    def __repr__(self):
        return f'ML potential (available = {self.is_available})'

    def generate_input_for(self, calc):
        """
        Just print a .xyz file of the molecule, which can be read
        as a gap-train configuration object.
        """

        calc.molecule.print_xyz_file(filename=calc.input.filename)
        calc.input.additional_filenames = [self.path]
        return None

    def output_filename_for(self, calc):
        return f'{calc.name}.xyz'

    def input_filename_for(self, calc):
        return f'{calc.name}.xyz'

    def version_in(self, calc):
        return '1.0.0'

    def execute(self, calc: Calculation):
        """
        Execute the calculation.
        Arguments:
            calc (autode.calculations.Calculation):
        """
        from ase.io.trajectory import Trajectory as ASETrajectory
        from ase.optimize import BFGS

        # @work_in_tmp_dir_ade(
        #     filenames_to_copy=calc.input.filenames,
        #     kept_file_exts=self.kept_file_exts,
        # )
        def execute_mlp():
            if 'opt' in self.action:
                # run optimisation
                logger.info('start optimization')
                ase_atoms = from_autode_to_ase(molecule=calc.molecule)
                logger.info('start optimise molecule')
                logger.info(f'{ase_atoms.cell}, {ase_atoms.pbc}')
                ase_atoms.calc = self.ase_calculator
                asetraj = ASETrajectory('tmp.traj', 'w', ase_atoms)
                dyn = BFGS(ase_atoms)
                dyn.attach(asetraj.write, interval=2)
                dyn.run(fmax=self.opt_fmax)
                traj = _convert_ase_traj('tmp.traj')
                final_traj = traj.final_frame
                final_traj.single_point(self.mlp, n_cores=calc.n_cores)
                name = self.output_filename_for(calc)
                final_traj.save_xyz(filename=name, predicted=True)

            else:
                # run single point calculation
                config_set = mlt.ConfigurationSet()
                config_set.load_xyz(
                    self.input_filename_for(calc), charge=0, mult=1
                )
                config = config_set[0]
                config.box = Box(size=[100, 100, 100])
                config.single_point(self.mlp, n_cores=calc.n_cores)
                name = self.output_filename_for(calc)
                config.save_xyz(filename=name, predicted=True)

        execute_mlp()
        return None

    def terminated_normally_in(self, calc):
        name = self.output_filename_for(calc)

        if os.path.exists(name):
            config_set = mlt.ConfigurationSet()
            config_set.load_xyz(name, charge=0, mult=1, load_energies=True)
            return config_set[0].energy.true is not None
        return False

    @property
    def ase_calculator(self):
        return self.mlp.ase_calculator

    def _energy_from(self, calc):
        name = self.output_filename_for(calc)
        config_set = mlt.ConfigurationSet()
        config_set.load_xyz(name, charge=0, mult=1, load_energies=True)
        config = config_set[0]
        energy = config.energy.true if config.energy.true is not None else config.energy.predicted
        return PotentialEnergy(
            energy * ev_to_ha, units='ha', method=self
        )

    def coordinates_from(self, calc):
        name = self.output_filename_for(calc)
        config_set = mlt.ConfigurationSet()
        config_set.load_xyz(name, charge=0, mult=1)
        config = config_set[0]
        return np.array([atom.coordinate for atom in config.atoms])

    def optimiser_from(self, calc):
        return None

    def get_free_energy(self, calc):
        return None

    def get_enthalpy(self, calc):
        return None

    def partial_charges_from(self, calc):
        return None

    def optimisation_converged(self, calc):
        return False

    def optimisation_nearly_converged(self, calc):
        return False

    def get_imaginary_freqs(self, calc):
        return None

    def get_normal_mode_displacements(self, calc, mode_number):
        return None

    def get_final_atoms(self, calc):
        name = self.output_filename_for(calc)
        config_set = mlt.ConfigurationSet()
        config_set.load_xyz(name, charge=0, mult=1)
        config = config_set[0]
        return [atom for atom in config.atoms]

    def get_atomic_charges(self, calc):
        return None

    def gradient_from(self, calc):
        name = self.output_filename_for(calc)
        config_set = mlt.ConfigurationSet()
        config_set.load_xyz(
            name, charge=0, mult=1, load_energies=True, load_forces=True
        )
        config = config_set[0]
        return config.forces.true * ev_to_ha

    def uses_external_io(self):
        return True
