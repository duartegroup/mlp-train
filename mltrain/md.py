import os
import numpy as np
import autode as ade
from typing import Optional
from mltrain.configurations import Configuration, Trajectory
from mltrain.config import Config
from mltrain.log import logger
from mltrain.box import Box
from mltrain.utils import work_in_tmp_dir
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory as ASETrajectory
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase import units as ase_units
from numpy.random import RandomState


@work_in_tmp_dir(copied_exts=['.xml', '.json', '.pth'])
def run_mlp_md(configuration: 'mltrain.Configuration',
               mlp:           'mltrain.potentials._base.MLPotential',
               temp:          float,
               dt:            float,
               interval:      int,
               init_temp:     Optional[float] = None,
               fbond_energy:  Optional[dict] = None,
               bbond_energy:  Optional[dict] = None,
               **kwargs
               ) -> 'mltrain.Trajectory':
    """
    Run molecular dynamics on a system using a MLP to predict energies and
    forces and ASE to drive dynamics

    ---------------------------------------------------------------------------
    Arguments:
        configuration:

        mlp:

        temp: Temperature in K to initialise velocities and to run
              NVT MD, if temp=0 then will run NVE

        init_temp: (float | None) Initial temperature to initialise momenta
                   with. If None then will be set to temp

        dt: (float) Time-step in fs

        interval: (int) Interval between saving the geometry

    -------------------
    Keyword Arguments:

        {fs, ps, ns}: Simulation time in some units

        bbond_energy (dict | None):  Additional energy to add to a breaking
                         bond. e.g. bbond_energy={(0, 1), 0.1} Adds 0.1 eV
                         to the 'bond' between atoms 0 and 1 as velocities
                         shared between the atoms in the breaking bond direction

        :fbond_energy (dict | None): As bbond_energy but in the direction to
                         form a bond

        n_cores (int): Number of cores to use

    Returns:
        (mltrain.ConfigurationSet):
    """
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
    ase_atoms.set_calculator(mlp.ase_calculator)

    _set_momenta(ase_atoms,
                 temp=init_temp if init_temp is not None else temp,
                 bbond_energy=bbond_energy,
                 fbond_energy=fbond_energy)

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

    dyn.attach(append_energy, interval=interval)
    dyn.attach(traj.write, interval=interval)

    logger.info(f'Running {n_steps:.0f} steps with a timestep of {dt} fs')
    dyn.run(steps=n_steps)

    traj = _convert_ase_traj('tmp.traj')

    for i, (frame, energy) in enumerate(zip(traj, energies)):
        frame.update_attr_from(configuration)
        frame.energy.predicted = energy
        frame.time = dt * interval * i

    return traj


def _convert_ase_traj(filename: str) -> 'mltrain.Trajectory':
    """Convert an ASE trajectory into a mltrain ConfigurationSet"""

    ase_traj = ASETrajectory(filename)
    mlt_traj = Trajectory()

    # Iterate through each frame (set of atoms) in the trajectory
    for atoms in ase_traj:
        config = Configuration()
        config.atoms = [ade.Atom(label) for label in atoms.symbols]

        # Set the coordinate of every atom in the configuration
        for i, position in enumerate(atoms.get_positions()):
            config.atoms[i].coord = position

        mlt_traj.append(config)

    return mlt_traj


def _set_momenta(ase_atoms:     'ase.atoms.Atoms',
                 temp:          float,
                 bbond_energy:  dict,
                 fbond_energy:  dict):
    """Set the initial momenta of some ASE atoms"""

    if temp > 0:
        logger.info(f'Initialising initial velocities for {temp} K')

        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temp,
                                     rng=RandomState())
    else:
        # Set the momenta to zero
        ase_atoms.arrays['momenta'] = np.zeros((len(ase_atoms), 3))

    def add_momenta(idx, vector, energy):
        masses = ase_atoms.get_masses()
        ase_atoms.arrays['momenta'][idx] = (np.sqrt(masses[idx] * energy) * vector)
        return None

    coords = ase_atoms.positions
    if bbond_energy is not None:
        logger.info('Adding breaking bond momenta')

        for atom_idxs, energy in bbond_energy.items():
            i, j = atom_idxs
            logger.info(f'Adding {energy} eV to break bond: {i}-{j}')

            #    vec
            #   <---   i--j         where i and j are two atoms
            #
            vec = coords[i] - coords[j]
            vec /= np.linalg.norm(vec)    # normalise

            add_momenta(idx=i, vector=vec, energy=energy)
            add_momenta(idx=j, vector=-vec, energy=energy)

    if fbond_energy is not None:
        for atom_idxs, energy in fbond_energy.items():
            i, j = atom_idxs
            logger.info(f'Adding {energy} eV to form bond: {i}-{j}')

            #    vec
            #   --->   i--j         where i and j are two atoms
            #
            vec = coords[j] - coords[i]
            vec /= np.linalg.norm(vec)  # normalise

            add_momenta(idx=i, vector=vec, energy=energy)
            add_momenta(idx=j, vector=-vec, energy=energy)

    return None


def _n_simulation_steps(dt: float,
                        kwargs: dict) -> int:
    """Calculate the number of simulation steps from a set of keyword
    arguments e.g. kwargs = {'fs': 100}

    ---------------------------------------------------------------------------
    Arguments:
        dt: Timestep in fs

        kwargs:

    Returns:
        (int): Number of simulation steps to perform
    """
    if dt < 0.09 or dt > 5:
        logger.warning('Unexpectedly small or large timestep - is it in fs?')

    if 'ps' in kwargs:
        time_fs = 1E3 * kwargs['ps']

    elif 'fs' in kwargs:
        time_fs = kwargs['fs']

    elif 'ns' in kwargs:
        time_fs = 1E6 * kwargs['ns']

    else:
        raise ValueError('Simulation time not found')

    n_steps = max(int(time_fs / dt), 1)                 # Run at least one step

    return n_steps
