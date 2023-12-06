import mlptrain as mlt
import numpy as np
from mlptrain.log import logger
from mlptrain.box import Box
from scipy.spatial import distance_matrix
import os
from generate_rs import generate_rs

mlt.Config.n_cores = 10


def grid_box(positions, size, grid_space):
    """to put grid box in the system"""
    size_x_min = np.min(positions.T[0])-size
    size_x_max = np.max(positions.T[0])+size
    size_y_min = np.min(positions.T[1])-size
    size_y_max = np.max(positions.T[1])+size
    size_z_min = np.min(positions.T[2])-size
    size_z_max = np.max(positions.T[2])+size

    x_space = np.arange(size_x_min, size_x_max, grid_space)
    y_space = np.arange(size_y_min, size_y_max, grid_space)
    z_space = np.arange(size_z_min, size_z_max, grid_space)

    grid_points = np.vstack(np.meshgrid(x_space,y_space,z_space)).reshape(3,-1).T
    return grid_points


def cavity_volume(ase_system, solute_idx = 22, grid_side_length = 0.2, radius = 1.5):
    """calcuate the cavity volume"""
    
    solute_sys = ase_system[:solute_idx]
    solvent_sys = ase_system[solute_idx:]

    grid_pos = grid_box (positions=solute_sys.get_positions(), size=2.0, grid_space=grid_side_length)

    volume = 0
    for point in grid_pos:
        dist_matrix = distance_matrix([point], solvent_sys.get_positions())
        volume += 1*(np.sum(dist_matrix < radius)==0)
    volume*=(grid_side_length ** 3)
    return volume


def from_ase_to_autode(atoms):
    from autode.atoms import Atom
    autode_atoms = []
    symbols = atoms.symbols

    for i in range(len(atoms)):
        autode_atoms.append(Atom(symbols[i],
                          x=atoms.positions[i][0],
                          y=atoms.positions[i][1],
                          z=atoms.positions[i][2]))

    return autode_atoms


@mlt.utils.work_in_tmp_dir(copied_exts=['.xml', '.json'])
def md_with_file(configuration, mlp, temp, dt, interval, init_temp = None, **kwargs):
    from mltrain.md import _convert_ase_traj, _n_simulation_steps
    from ase.io.trajectory import Trajectory as ASETrajectory
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet
    from ase import units as ase_units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from numpy.random import RandomState

    logger.info('Running MLP MD')

    # For modestly sized systems there is some slowdown using >8 cores
    n_cores = kwargs['n_cores'] if 'n_cores' in kwargs else min(mlt.Config.n_cores, 8)
    n_steps = _n_simulation_steps(dt, kwargs)

    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} cores for MLP MD')

    if mlp.requires_non_zero_box_size and configuration.box is None:
        logger.warning('Assuming vaccum simulation. Box size = 1000 nm^3')
        configuration.box = Box([100, 100, 100])

    ase_atoms = configuration.ase_atoms
    ase_atoms.set_calculator(mlp.ase_calculator)
    bias = mlt.Bias(zeta_func=mlt.AverageDistance((1,12), (6,11)), kappa=0.4, reference=1.6)
    ase_atoms.set_constraint(bias)
    MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temp,
                                     rng=RandomState())
    traj = ASETrajectory("tmp.traj", 'w', ase_atoms)
    
    energies = []
    def append_energy(_atoms = ase_atoms):
        energies.append(_atoms.get_potential_energy())
        
    reaction_coords = []
    def get_reaction_coord(atoms = ase_atoms):
        C2_C7 = np.linalg.norm(atoms[1].position-atoms[12].position)
        C4_C6 = np.linalg.norm(atoms[6].position-atoms[11].position)
        reaction_coord = 0.5*(C2_C7+C4_C6)
        reaction_coords.append(reaction_coord)

    cavity_volumn = []
    def get_cavity_volume(atoms = ase_atoms):
        volumn = cavity_volume(ase_system=atoms)
        cavity_volumn.append(volumn)
    if temp > 0:                                         # Default Langevin NVT
        dyn = Langevin(ase_atoms, dt * ase_units.fs,
                       temperature_K=temp,
                       friction=0.02)
    else:                                               # Otherwise NVE
        dyn = VelocityVerlet(ase_atoms, dt * ase_units.fs)

    dyn.attach(append_energy, interval=interval)
    dyn.attach(get_reaction_coord,interval=interval)
    dyn.attach(get_cavity_volume,interval=interval)
    dyn.attach(traj.write, interval=interval)

    logger.info(f'Running {n_steps:.0f} steps with a timestep of {dt} fs')
    dyn.run(steps=n_steps)

    traj = _convert_ase_traj('tmp.traj')

    for i, (frame, energy) in enumerate(zip(traj, energies)):
        frame.update_attr_from(configuration)
        frame.energy.predicted = energy
        frame.time = dt * interval * i

    return traj, reaction_coords, cavity_volumn


def traj_study(configs,  ml_potential,  init_md_time_fs = 500, max_time_fs = 3000):     
    
    C2_C7_initial_list = []
    C4_C6_initial_list = []

    for k in range(500):
        config =configs[k]
        logger.info(f'start trajectory study for {k} th configuration')

        C2_C7 = np.linalg.norm(config.atoms[1].coord-config.atoms[12].coord)
        C4_C6 = np.linalg.norm(config.atoms[6].coord-config.atoms[11].coord)

        C2_C7_initial_list.append(C2_C7)
        C4_C6_initial_list.append(C4_C6)

        C2_C7_list_f = []
        C4_C6_list_f = []

        C2_C7_list_f.append(C2_C7)
        C4_C6_list_f.append(C4_C6)

        tol_md_time_f = init_md_time_fs
        md_time_fs_f = init_md_time_fs

        while tol_md_time_f <= max_time_fs:
            C2_C7_list = []
            C4_C6_list = []

            traj, reaction_coords, cavity_volume = md_with_file(config,
                                                                mlp=ml_potential,
                                                                temp=300,
                                                                dt=0.5,
                                                                interval=2,
                                                                fs=md_time_fs_f)
            ending = 0
            for (i, j) in zip (C2_C7_list, C4_C6_list):
                logger.info(f'C2-C7 and C4-C6 bond lengths are {(i,j)}')
                if i <= 1.6 and j <= 1.6:
                    ending += 1
                    break
                else:
                    pass

            if ending!= 0:
                traj.save_xyz(f'trajectoris/traj_{k}.xyz')
                with open ('reaction_coords.txt','a') as f:
                    line = reaction_coords
                    print(line, file=f)
                with open ('cavity_volume.txt','a') as f:
                    line = cavity_volume
                    print(line, file=f)
                break

            config = traj[-1]
            md_time_fs_f = 1000
            tol_md_time_f += md_time_fs_f
            logger.info(f'current simulation time is {tol_md_time_f} fs')
    return None

if __name__ == '__main__':
    water_mol = mlt.Molecule(name='h2o.xyz')
    TS_mol = mlt.Molecule(name='cis_endo_TS_wB97M.xyz')

    system = mlt.System(TS_mol, box=Box([100, 100, 100]))
    system.add_molecules(water_mol, num=52)

    endo = mlt.potentials.ACE('endo_in_water_ace_wB97M', system)

    TS = mlt.ConfigurationSet()
    TS.load_xyz(filename = 'cis_endo_TS_wB97M.xyz', charge = 0, mult = 1)
    TS = TS[0]
    TS.box = Box([21.5, 21.5,21.5])
    TS.charge = 0
    TS.mult = 1

    water_system = mlt.System(water_mol, box=Box([21.5, 21.5,21.5]))
    water_system.add_molecules(water_mol, num=331)

    rs = generate_rs(TS, water_system, endo, 21.5)

    traj_study (rs, endo)
