"""
Helpers for producing PES plots of a given reactive system.
"""
import os
import mlptrain as mlt
from mlptrain.box import Box
import multiprocessing as mp
import autode as ade
from autode.utils import work_in_tmp_dir as work_in_tmp_dir_ade
from autode.wrappers.methods import ExternalMethodOEG
from autode.wrappers.keywords import KeywordsSet
from autode.calculations.calculation import Calculation
from autode.values import PotentialEnergy
from ase.constraints import Hookean
from ase.geometry import find_mic
from mlptrain.log import logger
from mlptrain.config import Config
from mlptrain.sampling.md import _convert_ase_traj
from mlptrain.utils import work_in_tmp_dir as work_in_tmp_dir_mlt
import numpy as np
from copy import deepcopy
import torch

ev_to_ha = 1.0 / 27.2114

# TODO: DEBUG, REMOVE THIS!!!
N_CORES = 1
mlt.Config.n_cores = N_CORES
ade.Config.n_cores = N_CORES

if torch.cuda.is_available():
    mlt.Config.mace_params['calc_device'] = 'cuda'


def adjust_potential_energy(self, atoms):
    """
    Returns the difference to the potential energy due to an active
    constraint. (That is, the quantity returned is to be added to the
    potential energy)
    """
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
    """
    Updates atomic forces due to an active constraint.
    """
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

        @work_in_tmp_dir_ade(
            filenames_to_copy=calc.input.filenames,
            kept_file_exts=self.kept_file_exts,
        )
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
        logger.info(f'True Energy: {config.energy.true}')
        return PotentialEnergy(
            config.energy.true * ev_to_ha, units='ha', method=self
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


@work_in_tmp_dir_mlt()
def optimise_with_fix_solute(
    config: mlt.Configuration,
    fmax: float,
    mlp: mlt.potentials.MLPotential,
    solute: mlt.Configuration = None,
    **kwargs,
) -> mlt.Configuration:
    """
    Optimise the configuration by MLP with a fixed solute (solute coords should be the first in configuration coords).

    Parameters:
        config (mlt.Configuration): the configuration either in vacuum or in solvent where the first len(solute) atoms
                                    are those of the solute.
        fmax (float):               fmax value for BFGS optimiser
        mlp (mlt.potentials.MLPotential):
        solute (mlt.Configuration): 'solute' configuration, if specified, takes the number of atoms in this config to determine
                                    the first n atoms of 'config' to fix with constraints.
    Returns:
        mlt.Configuration: final frame config of optimised trajectory.
    """
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

    # get ase atoms and load calculator
    ase_atoms = config.ase_atoms
    logger.info(f'{ase_atoms.cell}, {ase_atoms.pbc}')
    ase_atoms.calc = mlp.ase_calculator

    # constrain solute atoms if specified
    if solute is not None:
        solute_idx = list(range(len(solute.atoms)))
        constraints = FixAtoms(indices=solute_idx)
        ase_atoms.set_constraint(constraints)

    # run optimisation
    asetraj = ASETrajectory('tmp.traj', 'w', ase_atoms)
    dyn = BFGS(ase_atoms)
    dyn.attach(asetraj.write, interval=2)
    dyn.run(fmax=fmax)

    # return final optimisation trajectory frame
    traj = _convert_ase_traj('tmp.traj')
    final_traj = traj.final_frame
    return final_traj


def calculate_pes(
    mlp: mlt.potentials.MLPotential,
    ts_config: mlt.Configuration,
    react_coords: list[tuple],
    save_name: str,
    solvent_mol: mlt.Molecule = None,
    solvent_density: float = None,
    solvation_box_size: float = 14.0,
    opt_fmax: float = 0.01,
    grid_spec: tuple = (1.50, 3.5, 25),
    box_dim: tuple = (100.0, 100.0, 100.0),
    kept_file_exts: tuple[str] = ('.xyz'),
):
    """
    Calculates an n-dimensional potential energy surface (PES) for a given reactive system about
    a set of given reaction coordinates.
    Currently only tested for unimolecular (X -> A + B) and bimolecular (A + B -> X) reactions.

    Parameters:
        mlp (mlt.potentials.MLPotential): the machine learning potential model to use for calculation
        ts_xyz_fpath (str): the file path of the xyz file of the transition state of the reaction (in vacuum)
        reaction_coords (list(tuple): a list of reaction coordinates of pairs of atoms expected to form / break a bond
                                      i.e. [(1, 12), (6, 11)] means a bond formed between atom 1 and 12 and atom 6 and 11.
        save_name (str): save file name stub for all produced pes files
        solvent_xyz_fpath (str): xyz fpath of the solvent molecule
        solvent_density (float): experimental density (in g/cm^3) to try to replicate with solvent placement
        solvation_box_size (float): the size to fit the solvent to (controls number of solvent molecules placed,
                                    but does not define the final box size for PES scan)
        opt_fmax (float): the convergence value for the BFGS optimiser for geometry optimisation
        grid_spec (tuple): a tuple of (start_dist, end_dist, grid_size) where start_dist and end_dist specify the start and
                            ends of the distances between reaction coords in Amstrongs to perform the scan over a grid of grid_size^2.
        box_dim (tuple): box dimensions for running PES calc (default to 100, for non-periodic cluster)
        kept_file_exts (tuple[str]): tuple of file extensions to keep in directory after run (e.g. .xyz)
    """

    Hookean.adjust_forces = adjust_forces
    Hookean.adjust_potential_energy = adjust_potential_energy

    # set multi start method to avoid CUDA errors
    mp.set_start_method('spawn', force=True)

    ts_config.box = Box(box_dim)
    product_fname = f'{save_name}_product.xyz'

    # print true DFT TS location (in reaction coordinates)
    ts_rs_distances = [
        np.linalg.norm(ts_config.atoms[a1].coord - ts_config.atoms[a2].coord)
        for a1, a2 in react_coords
    ]
    logger.info(
        'Reference TS Reaction Distances: '
        + ', '.join(
            [
                f'r{i}: {ts_rs_distances[i]}'
                for i in range(len(ts_rs_distances))
            ]
        )
    )

    # 1) run biased MD to go from TS -> Product
    trajectory_product = mlt.md.run_mlp_md(
        configuration=ts_config,
        mlp=mlp,
        fs=500,
        temp=300,
        dt=0.5,
        fbond_energy={coord: 0.1 for coord in react_coords},
        interval=2,
    )
    final_traj_product = trajectory_product.final_frame

    # solvate the product
    if solvent_mol is not None:
        final_traj_product.solvate(
            box_size=solvation_box_size,
            solvent_molecule=solvent_mol,
            solvent_density=solvent_density,
        )
    final_traj_product.save_xyz('final_traj_product')
    final_traj_product.box = Box(box_dim)

    # 2) fix the solute and run optimisation of solvent (if there is any)
    traj_product_optimised = optimise_with_fix_solute(
        solute=ts_config,
        config=final_traj_product,
        fmax=opt_fmax,
        mlp=mlp,
        constraint=False,
    )

    # log product react coord distances
    prod_rs_distances = [
        np.linalg.norm(
            traj_product_optimised.atoms[a1].coord
            - traj_product_optimised.atoms[a2].coord
        )
        for a1, a2 in react_coords
    ]
    logger.info(
        'the forming carbon bonds length in product are: '
        + ', '.join(
            [
                f'r{i}: {prod_rs_distances[i]}'
                for i in range(len(prod_rs_distances))
            ]
        )
    )

    # save product to file
    product = mlt.Molecule(name='product', atoms=traj_product_optimised.atoms)
    product.print_xyz_file(filename=product_fname)

    # define PES
    pes = ade.pes.RelaxedPESnD(
        ade.Molecule(product_fname),
        rs={react_coord: grid_spec for react_coord in react_coords},
    )

    # define ade Method
    ade_mlp_method = MLPEST(
        mlp=mlp,
        action=['opt'],
        path=f'{cwd}/{mlp.name}.model',
        opt_fmax=opt_fmax,
        kept_file_exts=kept_file_exts,
    )

    # check if .npz file of PES already exists, and load
    npz_fname = f'{save_name}_pes.npz'
    if os.path.exists(npz_fname):
        pes.load(npz_fname)
    else:
        # calculate the pes
        pes.calculate(
            method=ade_mlp_method, keywords=['opt'], n_cores=mlt.Config.n_cores
        )

    # save npz of PES and save plot
    pes.save(filename=npz_fname)
    pes.plot(
        filename=f'{save_name}_pes.pdf',
    )


if __name__ == '__main__':

    solvent_density = None
    solvent_xyz_fpath = None

    # select model
    # model_name = 'GO-MACE-23'
    # model_name = 'MACE-MP0-128-L1'
    # model_name = 'MACE-OFF23_medium'
    model_name = 'MACE-OFF23_medium_endo_DA_train_fine_tuned'

    # select input TS and PES options
    ts_xyz_fpath = 'cis_endo_TS_wB97M.xyz'
    system_name = 'cis_endo_DA'
    react_coords = [(1, 12), (6, 11)]
    opt_fmax = 0.5
    box_dim = [100.0, 100.0, 100.0]
    grid_spec = (2.0, 2.2, 2)  # debug
    # grid_spec = (1.55, 3.0, 20)  # debug

    # select solvent params
    solvent_xyz_fpath, solvent_density = 'h2o.xyz', 0.99657
    solvation_box_size = 5.0
    
    # load model
    system = mlt.System(box=box_dim)
    cwd = os.getcwd()
    mlp = mlt.potentials.MACE(
        model_name, system, model_fpath=f'{cwd}/{model_name}.model'
    )

    # load ts config + solvent mol
    ts_set = mlt.ConfigurationSet()
    ts_set.load_xyz(filename=ts_xyz_fpath, charge=0, mult=1)
    ts_config = ts_set[0]
    solvent_mol = mlt.Molecule(solvent_xyz_fpath) if solvent_xyz_fpath is not None else None

    solvent_str = f"_in_{solvent_xyz_fpath.split('/')[-1].split('.')[0]}" if solvent_xyz_fpath is not None else ""
    save_name = f'{system_name}{solvent_str}_{model_name}'

    calculate_pes(
        mlp,
        ts_config,
        react_coords,
        save_name,
        solvent_mol=None,
        solvent_density=solvent_density,
        solvation_box_size=solvation_box_size,
        opt_fmax=opt_fmax,
        grid_spec=grid_spec,
        box_dim=box_dim,
        kept_file_exts=None,
    )
