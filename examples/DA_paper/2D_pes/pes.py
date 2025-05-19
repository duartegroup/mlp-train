import os
import mlptrain as mlt
from mlptrain.box import Box
import multiprocessing as mp
import autode as ade
from autode.utils import work_in_tmp_dir as work_in_tmp_dir_ade
from autode.wrappers.methods import ExternalMethodOEG
from autode.wrappers.keywords import KeywordsSet
from autode.calculations.types import CalculationType
from autode.calculations.calculation import Calculation
from autode.calculations.executors import CalculationExecutor
from ase.constraints import Hookean
from ase.geometry import find_mic
from mlptrain.log import logger
from mlptrain.config import Config
from mlptrain.sampling.md import _convert_ase_traj
from mlptrain.utils import work_in_tmp_dir as work_in_tmp_dir_mlt
import numpy as np
from copy import deepcopy

mlt.Config.n_cores = 1
ade.Config.n_cores = 1

# if torch.cuda.is_available():
mlt.Config.mace_params['calc_device'] = 'cuda'

ev_to_ha = 1.0 / 27.2114


def adjust_potential_energy(self, atoms):
    """Returns the difference to the potential energy due to an active
    constraint. (That is, the quantity returned is to be added to the
    potential energy)"""
    positions = atoms.positions
    if self._type == 'plane':
        A, B, C, D = self.plane
        x, y, z = positions[self.index]
        d = (A * x + B * y + C * z + D) / np.sqrt(A**2 + B**2 + C**2)
        if d > 0:
            return 0.5 * self.spring * d**2
        else:
            return 0.0

    if self._type == 'two atoms':
        p1, p2 = positions[self.indices]

    elif self._type == 'point':
        p1 = positions[self.index]
        p2 = self.origin
    displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
    bondlength = np.linalg.norm(displace)
    return 0.5 * self.spring * (bondlength - self.threshold) ** 2


def adjust_forces(self, atoms, forces):
    positions = atoms.positions
    if self._type == 'plane':
        A, B, C, D = self.plane
        x, y, z = positions[self.index]
        d = (A * x + B * y + C * z + D) / np.sqrt(A**2 + B**2 + C**2)
        if d < 0:
            return 0
        magnitude = self.spring * d
        direction = -np.array((A, B, C)) / np.linalg.norm((A, B, C))
        forces[self.index] += direction * magnitude
        return None

    if self._type == 'two atoms':
        p1, p2 = positions[self.indices]

    elif self._type == 'point':
        p1 = positions[self.index]
        p2 = self.origin
    displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
    bondlength = np.linalg.norm(displace)
    magnitude = self.spring * (bondlength - self.threshold)
    direction = displace / np.linalg.norm(displace)

    if self._type == 'two atoms':
        forces[self.indices[0]] += direction * magnitude
        forces[self.indices[1]] -= direction * magnitude

    else:
        forces[self.index] += direction * magnitude
    return None


def from_autode_to_ase(molecule, cell_size=100):
    """convert autode.molecule to ase.atoms
    maintain the constrain generated during ade.pes.RelaxedPESnD calculation"""
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
    """class of machine learning potential fitted for autode package
    original code provided by T. Yang"""

    @property
    def is_available(self):
        return True

    def __repr__(self):
        return f'ML potential (available = {self.is_available})'

    def generate_input_for(self, calc):
        """Just print a .xyz file of the molecule, which can be read
        as a gap-train  configuration object"""

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
        Execute the calculation
        Arguments:
            calc (autode.calculations.Calculation):
        """
        from ase.io.trajectory import Trajectory as ASETrajectory
        from ase.optimize import BFGS

        @work_in_tmp_dir_ade(
            filenames_to_copy=calc.input.filenames, kept_file_exts=('.xyz')
        )
        def execute_mlp():
            if 'opt' in self.action:
                logger.info('start optimization')
                ase_atoms = from_autode_to_ase(molecule=calc.molecule)
                logger.info('start optimise molecule')
                logger.info(f'{ase_atoms.cell}, {ase_atoms.pbc}')
                ase_atoms.calc = self.ase_calculator
                asetraj = ASETrajectory('tmp.traj', 'w', ase_atoms)
                dyn = BFGS(ase_atoms)
                dyn.attach(asetraj.write, interval=2)
                dyn.run(fmax=0.01)
                traj = _convert_ase_traj('tmp.traj')
                final_traj = traj.final_frame
                final_traj.single_point(self.mlp, n_cores=calc.n_cores)
                name = self.output_filename_for(calc)
                final_traj.save_xyz(filename=name, predicted=True)

            else:
                # run single point calculation
                config_set = mlt.ConfigurationSet()
                config_set.load_xyz(self.input_filename_for(calc), charge=0, mult=1)
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
        logger.info(f'True Energy: {config.energy.true}')
        logger.info(f'Pred Energy: {config.energy.predicted}')
        return config.energy.true * ev_to_ha

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
        config_set.load_xyz(name, charge=0, mult=1, load_energies=True, load_forces=True)
        config = config_set[0]
        return config.forces.true * ev_to_ha

    # def implements(self, calculation_type: CalculationType):
    #     return True

    def uses_external_io(self):
        return True

    def __init__(self, mlp, action, path):
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


def get_final_species(TS: mlt.Configuration, mlp: mlt.potentials.MLPotential):
    """get the optimised product after MD propogation"""
    trajectory_product = mlt.md.run_mlp_md(
        configuration=TS,
        mlp=mlp,
        fs=500,
        temp=300,
        dt=0.5,
        fbond_energy={(1, 12): 0.1, (6, 11): 0.1},
        interval=2,
    )

    final_traj_product = trajectory_product.final_frame

    traj_product_optimised = optimise_with_fix_solute(
        solute=TS,
        config=final_traj_product,
        fmax=0.01,
        mlp=mlp,
        constraint=False,
    )

    rt1 = np.linalg.norm(
        traj_product_optimised.atoms[1].coord
        - traj_product_optimised.atoms[12].coord
    )
    rt2 = np.linalg.norm(
        traj_product_optimised.atoms[6].coord
        - traj_product_optimised.atoms[11].coord
    )
    logger.info(f'the forming carbon bonds length in product are {rt1}, {rt2}')

    product = mlt.Molecule(name='product', atoms=traj_product_optimised.atoms)
    return product


@work_in_tmp_dir_mlt()
def optimise_with_fix_solute(solute, 
                             config: mlt.Configuration, 
                             fmax, 
                             mlp, 
                             constraint=True, 
                             **kwargs
):
    """optimised molecular geometries by MLP with or without constraint"""
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from ase.io.trajectory import Trajectory as ASETrajectory

    assert config.box is not None, 'configuration must have box'
    logger.info(
        'Optimise the configuration with fixed solute (solute coords should at the first in configuration coords) by MLP'
    )

    n_cores = (
        kwargs['n_cores'] if 'n_cores' in kwargs else min(Config.n_cores, 8)
    )
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} cores for MLP MD')

    ase_atoms = config.ase_atoms
    logger.info(f'{ase_atoms.cell}, {ase_atoms.pbc}')
    print('Current Working Dir: ', os.getcwd())
    ase_atoms.calc = mlp.ase_calculator

    if constraint:
        solute_idx = list(range(len(solute.atoms)))
        constraints = FixAtoms(indices=solute_idx)
        ase_atoms.set_constraint(constraints)

    asetraj = ASETrajectory('tmp.traj', 'w', ase_atoms)
    dyn = BFGS(ase_atoms)
    dyn.attach(asetraj.write, interval=2)
    dyn.run(fmax=fmax)

    traj = _convert_ase_traj('tmp.traj')
    final_traj = traj.final_frame
    return final_traj


Hookean.adjust_forces = adjust_forces
Hookean.adjust_potential_energy = adjust_potential_energy

if __name__ == '__main__':

    # set multi start method to avoid CUDA errors
    print(mp.get_start_method())
    mp.set_start_method("spawn", force=True)

    TS_mol = mlt.Molecule(name='cis_endo_TS_water.xyz')

    system = mlt.System(TS_mol, box=Box([100, 100, 100]))

    # endo = mlt.potentials.ACE('endo_ace_wB97M_imwater', system)
    # model_name = 'GO-MACE-23'
    # model_name = 'MACE-OFF23_medium'
    model_name = 'MACE-OFF23_medium_endo_DA_train_fine_tuned'
    cwd = os.getcwd()
    endo = mlt.potentials.MACE(model_name, system, model_fpath=f'{cwd}/{model_name}.model')

    # dumy_ase_calc = endo.ase_calculator

    # load transition state
    ts_set = mlt.ConfigurationSet()
    ts_set.load_xyz(filename='cis_endo_TS_wB97M.xyz', charge=0, mult=1)
    ts = ts_set[0]
    ts.box = Box([100, 100, 100])
    ts.charge = 0
    ts.mult = 1

    # ade_endo = MLPEST(mlp=endo, action=['opt'], path=f'{cwd}/{endo.name}.json')   # for ACE
    ade_endo = MLPEST(mlp=endo, action=['opt'], path=f'{cwd}/{endo.name}.model')

    product = get_final_species(ts, endo)

    product.print_xyz_file(filename='product.xyz')

    # define product
    pes = ade.pes.RelaxedPESnD(
        ade.Molecule('product.xyz'),
        rs={
            (1, 12): (1.55, 3, 20),  # Current->3.0 Å in 8 steps
            (6, 11): (1.55, 3, 20),
        },
    )

    # print key values
    # print('PES Energies: ', pes._energies)

    # calculate the pes
    pes.calculate(method=ade_endo, keywords=['opt'], n_cores=mlt.Config.n_cores)
    pes.save(filename='endo_in_water.npz')
    pes.plot()
