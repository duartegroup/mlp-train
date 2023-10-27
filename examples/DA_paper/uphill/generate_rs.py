import os
import mlptrain as mlt
from mlptrain.box import Box
from mlptrain.log import logger
from mlptrain.config import Config
from mlptrain.md import  _convert_ase_traj
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from numpy.random import RandomState
import numpy as np

mlt.Config.n_cores = 10

ev_to_ha = 1.0 / 27.2114


def from_ase_to_autode(atoms):
    from autode.atoms import Atom
    #atoms is ase.Atoms
    autode_atoms = []
    symbols = atoms.symbols

    for i in range(len(atoms)):
        autode_atoms.append(Atom(symbols[i],
                          x= atoms.positions[i][0],
                          y=atoms.positions[i][1],
                          z=atoms.positions[i][2]))

    return autode_atoms


def solvation(solute_config, solvent_config, apm, radius, enforce = True):
    """same function applied in training an MLP for reaction in explicit water
       solute: mlt.Configuration() solute.box is not None
       solvent: mlt.Configuration() solvent.box is not None
       aps: number of atoms per solvent molecule
       radius: cutout radius around each solute atom
       enforce: True / False Wrap solvent regardless of previous solvent PBC choices"""
    assert solute_config.box is not None, 'configuration must have box'
    assert solvent_config.box is not None, 'configuration must have box'

    solute = solute_config.ase_atoms
    solvent = solvent_config.ase_atoms

    def wrap(D, cell, pbc):
        """ wrap distance to nearest neighbor
            D: distance"""
        for i , periodic in enumerate(pbc):
            if periodic:
                d = D[:, i]
                L = cell[i]
                d[:] = (d+L/2)%L-L/2
        return None

    def molwrap(atoms, n, idx = 0):
        """Wrap to cell without breaking molecule
           n: number of atoms per solvent molecule
           idx: which atom in the solvent molecule to determine molecular distances from"""
        center = atoms.cell.diagonal()/2
        positions = atoms.positions.reshape((-1, n, 3))
        distances = positions[:, idx]-center
        old_distances = distances.copy()
        wrap(distances, atoms.cell.diagonal(), atoms.pbc)
        offsets = distances - old_distances
        positions += offsets[:, None]
        atoms.set_positions(positions.reshape((-1,3)))
        return atoms

    assert not (solvent.cell.diagonal()==0).any(), \
        'solvent atoms have no cell'
    assert (solvent.cell == np.diag(solvent.cell.diagonal())).all(), \
        'sol cell not orthorhombic'
    if enforce:
        solvent.pbc = True

    sol = molwrap(solvent, apm)

    solute.set_cell(sol.cell)
    solute.center()

    sys = solute + sol
    sys.pbc = True

    solute_idx = range(len(solute))
    mask = np.zeros(len(sys), bool)
    mask[solute_idx] = True

    atoms_to_delete = []
    for atm in solute_idx:
        if sys.numbers[atm] == 1:
            mol_dists = sys[atm].position - sys[~mask][::].positions
            idx = np.where((np.linalg.norm(mol_dists , axis = 1))<1.2)
            for i in idx:
                list_idx = i.tolist()
            for i in list_idx:
                n = i % apm
                for j  in range (apm-n):
                    atoms_to_delete.append(i+j+len(solute_idx))
                for j in range (n+1):
                    atoms_to_delete.append(i-j+len(solute_idx))
        elif sys.numbers[atm] == 8:
            mol_dists = sys[atm].position - sys[~mask][::].positions
            idx = np.where((np.linalg.norm(mol_dists , axis = 1))<1.52)
            for i in idx:
                list_idx = i.tolist()
            for i in list_idx:
                n = i % apm
                for j  in range (apm-n):
                    atoms_to_delete.append(i+j+len(solute_idx))
                for j in range (n+1):
                    atoms_to_delete.append(i-j+len(solute_idx))
        else:
            mol_dists = sys[atm].position - sys[~mask][::].positions
            idx = np.where((np.linalg.norm(mol_dists , axis = 1))<radius)
            for i in idx:
                list_idx = i.tolist()
            for i in list_idx:
                n = i % apm
                for j  in range (apm-n):
                    atoms_to_delete.append(i+j+len(solute_idx))
                for j in range (n+1):
                    atoms_to_delete.append(i-j+len(solute_idx))

    atoms_to_delete = np.unique(atoms_to_delete)
    del sys[atoms_to_delete]

    cell = sys.cell[:]
    autode_atoms = from_ase_to_autode(atoms=sys)
    solvation = mlt.Configuration(atoms=autode_atoms)
    solvation.box = Box([cell[0][0], cell[1][1], cell[2][2]])
    return solvation


@mlt.utils.work_in_tmp_dir(copied_exts=['.xml', '.json'])
def mlpmd_fix_solute(solute, configuration, mlp, temp, dt, interval, n_steps, **kwargs):
    """ run MLP MD with fixed solute atoms"""
    from ase.constraints import FixAtoms
    from ase.io.trajectory import Trajectory as ASETrajectory
    from ase.md.langevin import Langevin
    from ase import units as ase_units
    assert configuration.box is not None, 'configuration must have box'

    logger.info('Run MLP MD with fixed solute (solute coords should at the first in configuration coords) by MLP')

    n_cores = kwargs['n_cores'] if 'n_cores' in kwargs else min(Config.n_cores, 8)
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} cores for MLP MD')

    ase_atoms = configuration.ase_atoms
    logger.info(f'{ase_atoms.cell}, {ase_atoms.pbc}')
    ase_atoms.set_calculator(mlp.ase_calculator)

    solute_idx = list(range(len(solute.atoms)))
    constraints = FixAtoms(indices=solute_idx)
    ase_atoms.set_constraint(constraints)

    MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temp,
                                     rng=RandomState())
    
    asetraj = ASETrajectory("tmp.traj", 'w', ase_atoms)

    dyn = Langevin(ase_atoms, dt * ase_units.fs,
                       temperature_K=temp,
                       friction=0.02)

    dyn.attach(asetraj.write, interval=interval)

    logger.info(f'Running {n_steps:.0f} steps with a timestep of {dt} fs')
    dyn.run(steps=n_steps)

    traj = _convert_ase_traj('tmp.traj')
    return traj


@mlt.utils.work_in_tmp_dir(copied_exts=['.xml', '.json'])
def optimize_sys(configuration, mlp, **kwargs):
    # applied MLP to optimised geometry with BFGS method
    from ase.io.trajectory import Trajectory as ASETrajectory
    from ase.optimize import BFGS
    assert configuration.box is not None, 'configuration must have box'

    logger.info('Optimise the configuratoin with fixed solute (solute coords should at the first in configuration coords) by MLP')

    n_cores = kwargs['n_cores'] if 'n_cores' in kwargs else min(Config.n_cores, 8)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    #os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} cores for MLP MD')

    ase_atoms = configuration.ase_atoms
    logger.info(f'{ase_atoms.cell}, {ase_atoms.pbc}')
    ase_atoms.set_calculator(mlp.ase_calculator)
    asetraj = ASETrajectory("tmp.traj", 'w', ase_atoms)

    dyn = BFGS(ase_atoms)
    dyn.attach(asetraj.write, interval=2)
    dyn.run(fmax=0.05)

    traj = _convert_ase_traj('tmp.traj')
    final_traj = traj.final_frame
    return final_traj


def get_reactant_states(TS, solution, mlp):
    """ get RS by the following step:
        1) solvated TS in solution
        2) equilibrium solvent molecules by running MLP-MD for 20 ps
        3) Run MLP-MD initialized from the last frame 
           of equilibrated trajectory obtaiend from step 2) 
           adding a momtum to force the MD propogate to RS
        4) optimized thelast frame of trjactory obtained from step 3)"""  
    solved_in_solution = solvation (solute_config=TS,
                                    solvent_config=solution,
                                    apm=3,
                                    radius=1.7)

    after_fixed_md = mlpmd_fix_solute(solute=TS,
                                    configuration=solved_in_solution,
                                    mlp=mlp,
                                    temp=300,
                                    dt=0.5,
                                    interval=2,
                                    n_steps=40000)

    trajectory_reactant = mlt.md.run_mlp_md(configuration=after_fixed_md[-1],
                                       mlp=mlp,
                                       fs=200,
                                       temp=300,
                                       dt=0.5,
                                       bbond_energy ={(1,12) : 0.04, (6,11) : 0.04},
                                       interval=2)

    final_traj_reactant = trajectory_reactant.final_frame

    opt_reactant = optimize_sys(configuration=final_traj_reactant, mlp = mlp)

    rt1 = np.linalg.norm(opt_reactant.atoms[1].coord-opt_reactant.atoms[12].coord)
    rt2 = np.linalg.norm(opt_reactant.atoms[6].coord-opt_reactant.atoms[11].coord)
    logger.info(f'the forming carbon bonds length in reactant are {rt1}, {rt2}')
    return opt_reactant  


@mlt.utils.work_in_tmp_dir(copied_exts=['.xml', '.json', '.pth'])
def baised_md(configuration, mlp, temp, dt, interval, bias, **kwargs):
    from mltrain.md import _convert_ase_traj, _n_simulation_steps
    from ase.io.trajectory import Trajectory as ASETrajectory
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet
    from ase import units as ase_units
    logger.info('Running MLP MD')

    # For modestly sized systems there is some slowdown using >8 cores
    n_cores = kwargs['n_cores'] if 'n_cores' in kwargs else min(Config.n_cores, 8)
    n_steps = _n_simulation_steps(dt, kwargs)

    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} cores for MLP MD')

    if mlp.requires_non_zero_box_size and configuration.box is None:
        logger.warning('Assuming vaccum simulation. Box size = 1000 nm^3')
        configuration.box = Box([100, 100, 100])

    ase_atoms = configuration.ase_atoms
    logger.info(f'{ase_atoms.cell}, {ase_atoms.pbc}')
    ase_atoms.set_calculator(mlp.ase_calculator)
    ase_atoms.set_constraint(bias)

    MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temp,
                                     rng=RandomState())
    
    traj = ASETrajectory("tmp.traj", 'w', ase_atoms)
    energies = []

    def append_energy(_atoms=ase_atoms):
        energies.append(_atoms.get_potential_energy())

    if temp > 0:                                         # Default Langevin NVT
        dyn = Langevin(ase_atoms, dt * ase_units.fs,
                       temperature_K=temp,
                       friction=0.02)
    else:                                               # Otherwise NVE
        dyn = VelocityVerlet(ase_atoms, dt * ase_units.fs)
      
    dyn.attach(traj.write, interval=interval)

    logger.info(f'Running {n_steps:.0f} steps with a timestep of {dt} fs')
    dyn.run(steps=n_steps)
    traj = _convert_ase_traj('tmp.traj')

    trajectory = mlt.ConfigurationSet()
    for i in range(10, len(traj)):
        trajectory.append(traj[i])
    energies = energies[10:]
  
    for i, (frame, energy) in enumerate(zip(trajectory, energies)):
        frame.update_attr_from(configuration)
        frame.energy.predicted = energy
        frame.time = dt * interval * i
    return trajectory  


def generate_rs(TS, solution, mlp, box_size):
    ref = []
    reactants = mlt.ConfigurationSet()
    while len(reactants) < 10:
        reactant = get_reactant_states(TS=TS,
                                   solution=solution,
                                    mlp=mlp)
        rt1 = np.linalg.norm(reactant.atoms[1].coord-reactant.atoms[12].coord)
        rt2 = np.linalg.norm(reactant.atoms[6].coord-reactant.atoms[11].coord)
        if 3<rt1 <= 5 and 3<rt2 <= 5:
            reactants.append(reactant)
            ref.append(0.5*(rt1+rt2))

    for frame in reactants:
        frame.box = Box([box_size, box_size, box_size])

    rs = mlt.ConfigurationSet()
    for i, species in enumerate(reactants):
        bias = mlt.Bias(zeta_func=mlt.AverageDistance((1,12), (6,11)), kappa=0.5, reference=ref[i])
        traj = baised_md(configuration=species,
                              mlp=mlp,
                              temp=300,
                              dt=0.5,
                              interval=20,
                              fs=1000,
                              bias=bias)
        for step in traj:
            rt1 = np.linalg.norm(step.atoms[1].coord-step.atoms[12].coord)
            rt2 = np.linalg.norm(step.atoms[6].coord-step.atoms[11].coord)
            if 3 < rt1 <= 5 and 3 < rt2 <= 5:
                rs.append(step)              
    return rs
  
      
