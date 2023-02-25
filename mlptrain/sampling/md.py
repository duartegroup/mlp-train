import os
import numpy as np
import autode as ade
from typing import Optional, Union, List
from numpy.random import RandomState
from mlptrain.configurations import Configuration, Trajectory
from mlptrain.config import Config
from mlptrain.sampling import Bias, PlumedBias
from mlptrain.log import logger
from mlptrain.box import Box
from mlptrain.utils import work_in_tmp_dir
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory as ASETrajectory
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase import units as ase_units


@work_in_tmp_dir(copied_exts=['.xml', '.json', '.pth'],
                 kept_exts=['.dat', '.log'])
def run_mlp_md(configuration: 'mlptrain.Configuration',
               mlp:           'mlptrain.potentials._base.MLPotential',
               temp:          float,
               dt:            float,
               interval:      int,
               init_temp:     Optional[float] = None,
               fbond_energy:  Optional[dict] = None,
               bbond_energy:  Optional[dict] = None,
               bias:          Optional[Union['mlptrain.Bias',
                                             'mlptrain.PlumedBias']] = None,
               **kwargs
               ) -> 'mlptrain.Trajectory':
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


        bbond_energy (dict | None):  Additional energy to add to a breaking
                         bond. e.g. bbond_energy={(0, 1), 0.1} Adds 0.1 eV
                         to the 'bond' between atoms 0 and 1 as velocities
                         shared between the atoms in the breaking bond direction

        fbond_energy (dict | None): As bbond_energy but in the direction to
                         form a bond

        bias (mlptrain.Bias | mlptrain.PlumedBias): ASE or PLUMED constraint
                                                    to use in the dynamics

    ---------------
    Keyword Arguments:

        {fs, ps, ns}: Simulation time in some units

    Returns:
        (mlptrain.ConfigurationSet):
    """

    logger.info('Running MLP MD')

    n_cores = (kwargs['n_cores'] if 'n_cores' in kwargs
               else min(Config.n_cores, 8))

    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} core(s) for MLP MD')

    n_steps = _n_simulation_steps(dt, kwargs)
    # Transform dt from fs into ASE time units (for dynamics only)
    dt_ase = dt * ase_units.fs

    if mlp.requires_non_zero_box_size and configuration.box is None:
        logger.warning('Assuming vaccum simulation. Box size = 1000 nm^3')
        configuration.box = Box([100, 100, 100])

    ase_atoms = configuration.ase_atoms

    _set_momenta(ase_atoms,
                 temp=init_temp if init_temp is not None else temp,
                 bbond_energy=bbond_energy,
                 fbond_energy=fbond_energy)

    traj = ASETrajectory("tmp.traj", 'w', ase_atoms)
    energies = []

    def append_energy(_atoms=ase_atoms):
        energies.append(_atoms.get_potential_energy())

    if temp > 0:                                         # Default Langevin NVT
        dyn = Langevin(ase_atoms, dt_ase,
                       temperature_K=temp,
                       friction=0.02)
    else:                                               # Otherwise NVE
        dyn = VelocityVerlet(ase_atoms, dt_ase)

    dyn.attach(append_energy, interval=interval)
    dyn.attach(traj.write, interval=interval)

    logger.info(f'Running {n_steps:.0f} steps with a timestep of {dt} fs')

    if isinstance(bias, PlumedBias):
        logger.info('Using PLUMED bias for MLP MD')

        from ase.calculators.plumed import Plumed

        setup = _write_plumed_setup(bias, interval, **kwargs)

        if '_idx' in kwargs:
            logfile = f'plumed_{kwargs["_idx"]}.log'

        else:
            logfile = 'plumed.log'

        plumed_calc = Plumed(calc=mlp.ase_calculator,
                             input=setup,
                             log=logfile,
                             timestep=dt_ase,
                             atoms=ase_atoms,
                             kT=temp * ase_units.kB)
        ase_atoms.calc = plumed_calc

        dyn.run(steps=n_steps)
        plumed_calc.plumed.finalize()

    elif isinstance(bias, Bias):
        logger.info('Using ASE bias for MLP MD')

        ase_atoms.calc = mlp.ase_calculator
        ase_atoms.set_constraint(bias)

        dyn.run(steps=n_steps)

    else:
        dyn.run(steps=n_steps)

    traj = _convert_ase_traj('tmp.traj')

    for i, (frame, energy) in enumerate(zip(traj, energies)):
        frame.update_attr_from(configuration)
        frame.energy.predicted = energy
        frame.time = dt * interval * i

    return traj


def _convert_ase_traj(filename: str) -> 'mlptrain.Trajectory':
    """Convert an ASE trajectory into an mlptrain Trajectory"""

    ase_traj = ASETrajectory(filename)
    mlt_traj = Trajectory()

    # Iterate through each frame (set of atoms) in the trajectory
    for atoms in ase_traj:
        config = Configuration()
        config.atoms = [ade.Atom(label) for label in atoms.symbols]

        # Set the atom_pair_list of every atom in the configuration
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


def _write_plumed_setup(bias, interval, **kwargs) -> List:
    """Generate a list which represents the PLUMED input file"""

    setup = []

    # Converting PLUMED units to ASE units
    time_conversion = 1 / (ase_units.fs * 1000)
    energy_conversion = ase_units.mol / ase_units.kJ
    units_setup = ['UNITS '
                   'LENGTH=A '
                   f'TIME={time_conversion} '
                   f'ENERGY={energy_conversion}']

    if bias.setup is not None:
        setup = bias.setup

        if 'UNITS' in setup[0]:
            logger.info('Setting PLUMED units to ASE units')
            setup[0] = units_setup[0]

            return setup

        else:
            logger.warning('Unit conversion not found in PLUMED input file, '
                           'adding a conversion from PLUMED units to ASE units')
            setup.insert(0, units_setup[0])

            return setup

    setup.extend(units_setup)

    # Defining DOFs and CVs
    for cv in bias.cvs:
        setup.extend(cv.setup)

    # Metadynamics
    if '_method' in kwargs and kwargs['_method'] == 'metadynamics':
        hills_filename = f'HILLS_{kwargs["_idx"]}.dat'

        if bias.biasfactor is not None:
            biasfactor_setup = f'BIASFACTOR={bias.biasfactor} '

        else:
            biasfactor_setup = ''

        metad_setup = ['METAD '
                       f'ARG={bias.cv_sequence} '
                       f'PACE={bias.pace} '
                       f'HEIGHT={bias.height} '
                       f'SIGMA={bias.width_sequence} '
                       f'{biasfactor_setup}'
                       f'FILE={hills_filename}']
        setup.extend(metad_setup)

    # Printing trajectory in terms of DOFs and CVs
    for cv in bias.cvs:

        if cv.dof_names is not None:
            args = f'{cv.name},{cv.dof_sequence}'

        else:
            args = cv.name

        name_without_dot = '_'.join(cv.name.split('.'))

        if '_idx' in kwargs:
            colvar_filename = f'colvar_{name_without_dot}_{kwargs["_idx"]}.dat'

        else:
            colvar_filename = f'colvar_{name_without_dot}.dat'

        print_setup = ['PRINT '
                       f'ARG={args} '
                       f'FILE={colvar_filename} '
                       f'STRIDE={interval}']
        setup.extend(print_setup)

    return setup
