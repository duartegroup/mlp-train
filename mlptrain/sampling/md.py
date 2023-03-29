import os
import shutil
import numpy as np
import autode as ade
from copy import deepcopy
from typing import Optional, Sequence, List
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
from ase.io import read
from ase import units as ase_units


def run_mlp_md(configuration:      'mlptrain.Configuration',
               mlp:                'mlptrain.potentials._base.MLPotential',
               temp:               float,
               dt:                 float,
               interval:           int,
               init_temp:          Optional[float] = None,
               fbond_energy:       Optional[dict] = None,
               bbond_energy:       Optional[dict] = None,
               bias:               Optional = None,
               restart_files:      Optional[List[str]] = None,
               copied_substrings:  Sequence[str] = ('.xml', '.json', '.pth'),
               kept_substrings:    Optional[Sequence[str]] = None,
               **kwargs
               ) -> 'mlptrain.Trajectory':
    """
    Run molecular dynamics on a system using a MLP to predict energies and
    forces and ASE to drive dynamics. The function is executed in a temporary
    directory.

    ---------------------------------------------------------------------------
    Arguments:

        configuration: Configuration from which the simulation is started
                       (if restart is False)

        mlp: Machine learnt potential

        temp: Temperature in K to initialise velocities and to run
              NVT MD, if temp=0 then will run NVE

        init_temp: (float | None) Initial temperature to initialise momenta
                   with. If None then will be set to temp

        dt: (float) Time-step in fs

        interval: (int) Interval between saving the geometry

        bbond_energy: (dict | None) Additional energy to add to a breaking
                         bond. e.g. bbond_energy={(0, 1), 0.1} Adds 0.1 eV
                         to the 'bond' between atoms 0 and 1 as velocities
                         shared between the atoms in the breaking bond direction

        fbond_energy: (dict | None) As bbond_energy but in the direction to
                         form a bond

        bias: (mlptrain.Bias | mlptrain.PlumedBias) ASE or PLUMED constraint
              to use in the dynamics

        restart_files: List of files which are needed for restarting the
                       simulation

        kept_substrings: List of substrings with which files are copied back
                         from the temporary directory
                         e.g. '.json', 'trajectory_1.traj'

        copied_substrings: List of substrings with which files are copied
                           to the temporary directory
    ---------------
    Keyword Arguments:

        {fs, ps, ns}: Simulation time in some units

        {save_fs, save_ps, save_ns}: Trajectory saving interval in some units

        constraints: (List) List of ASE constraints to use in the dynamics

        write_plumed_setup: (bool) If True saves the PLUMED input file as
                            plumed_setup.dat

    Returns:

        (mlptrain.Trajectory):
    """

    restart = restart_files is not None

    if kept_substrings is None:
        kept_substrings = []

    copied_substrings_list = list(copied_substrings)
    kept_substrings_list = list(kept_substrings)

    if restart:
        logger.info('Restarting MLP MD')

        if not isinstance(restart_files, list):
            raise TypeError('Restart files must be a list')

        for file in restart_files:
            if not isinstance(file, str):
                raise TypeError('Restart files must be a list of strings '
                                'specifying filenames')

        if not any(file.endswith('.traj') for file in restart_files):

            raise ValueError('Restaring a simulation requires a .traj file '
                             'from the previous simulation')

        if (isinstance(bias, PlumedBias) and
                not any(file.endswith('.dat') for file in restart_files)):

            raise ValueError('Restarting a PLUMED simulation requires a '
                             'colvar.dat (and in the case of metadynamics also '
                             'a HILLS.dat) file from the previous simulation')

        copied_substrings_list.extend(restart_files)
        kept_substrings_list.extend(restart_files)

    else:
        logger.info('Running MLP MD')

    decorator = work_in_tmp_dir(copied_substrings=copied_substrings_list,
                                kept_substrings=kept_substrings_list)

    _run_mlp_md_decorated = decorator(_run_mlp_md)

    traj = _run_mlp_md_decorated(configuration, mlp, temp, dt, interval,
                                 init_temp, fbond_energy, bbond_energy,
                                 bias, restart_files, **kwargs)

    return traj


def _run_mlp_md(configuration:  'mlptrain.Configuration',
                mlp:            'mlptrain.potentials._base.MLPotential',
                temp:           float,
                dt:             float,
                interval:       int,
                init_temp:      Optional[float] = None,
                fbond_energy:   Optional[dict] = None,
                bbond_energy:   Optional[dict] = None,
                bias:           Optional = None,
                restart_files:  Optional[List[str]] = None,
                **kwargs
                ) -> 'mlptrain.Trajectory':
    """
    Run molecular dynamics on a system using a MLP to predict energies and
    forces and ASE to drive dynamics.

    ---------------------------------------------------------------------------
    Arguments:

        configuration: Configuration from which the simulation is started
                       (if restart is False)

        mlp: Machine learnt potential

        temp: Temperature in K to initialise velocities and to run
              NVT MD, if temp=0 then will run NVE

        init_temp: (float | None) Initial temperature to initialise momenta
                   with. If None then will be set to temp

        dt: (float) Time-step in fs

        interval: (int) Interval between saving the geometry

        bbond_energy: (dict | None) Additional energy to add to a breaking
                         bond. e.g. bbond_energy={(0, 1), 0.1} Adds 0.1 eV
                         to the 'bond' between atoms 0 and 1 as velocities
                         shared between the atoms in the breaking bond direction

        fbond_energy: (dict | None) As bbond_energy but in the direction to
                         form a bond

        bias: (mlptrain.Bias | mlptrain.PlumedBias) ASE or PLUMED constraint
              to use in the dynamics

        restart_files: List of files which are needed for restarting the
                       simulation
    ---------------
    Keyword Arguments:

        {fs, ps, ns}: Simulation time in some units

        {save_fs, save_ps, save_ns}: Trajectory saving interval in some units

        constraints: (List) List of ASE constraints to use in the dynamics

        write_plumed_setup: (bool) If True saves the PLUMED input file as
                            plumed_setup.dat

    Returns:

        (mlptrain.Trajectory):
    """

    restart = restart_files is not None

    n_cores = (kwargs['n_cores'] if 'n_cores' in kwargs
               else min(Config.n_cores, 8))

    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} core(s) for MLP MD')

    # Transform dt from fs into ASE time units (for dynamics only)
    dt_ase = dt * ase_units.fs
    n_steps = _n_simulation_steps(dt, kwargs)

    if restart and n_steps % interval != 0:
        raise NotImplementedError('Current implementation requires the number '
                                  'of steps to be divisible by the interval '
                                  'if the simulation is restarted')

    if mlp.requires_non_zero_box_size and configuration.box is None:
        logger.warning('Assuming vaccum simulation. Box size = 1000 nm^3')
        configuration.box = Box([100, 100, 100])

    ase_atoms = configuration.ase_atoms
    traj_name = _get_traj_name(restart_files, **kwargs)

    _set_momenta_and_geometry(ase_atoms,
                              temp=init_temp if init_temp is not None else temp,
                              bbond_energy=bbond_energy,
                              fbond_energy=fbond_energy,
                              restart_files=restart_files,
                              traj_name=traj_name)

    ase_traj = _initialise_traj(ase_atoms, restart_files, traj_name)

    n_previous_steps = interval * len(ase_traj)
    energies = [None for _ in range(len(ase_traj))]

    calculator = _attach_calculator_with_bias(ase_atoms, mlp, bias, temp,
                                              interval, dt_ase, restart,
                                              n_previous_steps, **kwargs)

    _run_dynamics(ase_atoms, ase_traj, traj_name, interval, temp, dt, dt_ase,
                  n_steps, energies, calculator, bias, **kwargs)

    traj = _convert_ase_traj(traj_name, bias, **kwargs)

    for i, (frame, energy) in enumerate(zip(traj, energies)):
        frame.update_attr_from(configuration)
        frame.energy.predicted = energy
        frame.time = dt * interval * i

    return traj


def _attach_calculator_with_bias(ase_atoms, mlp, bias, temp, interval, dt_ase,
                                 restart, n_previous_steps, **kwargs):
    """Sets up the calculator, attaches it to the ase_atoms together with a
    bias and constraints, and returns the final calculator"""

    if isinstance(bias, PlumedBias):
        logger.info('Using PLUMED bias for MLP MD')

        from ase.calculators.plumed import Plumed

        setup = _plumed_setup(bias, temp, interval, **kwargs)

        plumed_calc = Plumed(calc=mlp.ase_calculator,
                             input=setup,
                             timestep=dt_ase,
                             atoms=ase_atoms,
                             kT=temp*ase_units.kB,
                             restart=restart)

        if restart:
            plumed_calc.istep = n_previous_steps

        if bias.cvs is not None:
            for cv in bias.cvs:
                if cv.files is not None:
                    cv.write_files()

        ase_atoms.calc = plumed_calc

        if 'constraints' in kwargs and kwargs['constraints'] is not None:
            ase_atoms.set_constraint(kwargs['constraints'])

        return plumed_calc

    elif isinstance(bias, Bias):
        logger.info('Using ASE bias for MLP MD')

        ase_atoms.calc = mlp.ase_calculator

        if 'constraints' in kwargs and kwargs['constraints'] is not None:
            constraints_with_bias = deepcopy(kwargs['constraints'])
            constraints_with_bias.append(bias)

            ase_atoms.set_constraint(constraints_with_bias)

        else:
            ase_atoms.set_constraint(bias)

        return mlp.ase_calculator

    else:
        ase_atoms.calc = mlp.ase_calculator

        if 'constraints' in kwargs and kwargs['constraints'] is not None:
            ase_atoms.set_constraint(kwargs['constraints'])

        return mlp.ase_calculator


def _run_dynamics(ase_atoms, ase_traj, traj_name, interval, temp, dt, dt_ase,
                  n_steps, energies, calculator, bias, **kwargs) -> None:
    """Initialises dynamics object and runs dynamics"""

    if temp > 0:                                        # Default Langevin NVT
        dyn = Langevin(ase_atoms, dt_ase,
                       temperature_K=temp,
                       friction=0.02)
    else:                                               # Otherwise NVE
        dyn = VelocityVerlet(ase_atoms, dt_ase)

    def append_unbiased_energy():
        _append_unbiased_energy(ase_atoms, energies, calculator, bias)

    def save_trajectory():
        _save_trajectory(ase_traj, traj_name, **kwargs)

    dyn.attach(append_unbiased_energy, interval=interval)
    dyn.attach(ase_traj.write, interval=interval)

    if any(key in kwargs for key in ['save_fs', 'save_ps', 'save_ns']):
        dyn.attach(save_trajectory,
                   interval=_traj_saving_interval(dt, kwargs))

    logger.info(f'Running {n_steps:.0f} steps with a timestep of {dt} fs')
    dyn.run(steps=n_steps)

    # The calling process waits until PLUMED process has finished
    if isinstance(bias, PlumedBias):
        calculator.plumed.finalize()

    return None


def _append_unbiased_energy(ase_atoms, energies, calculator, bias) -> None:
    """Appends unbiased energy (biased MLP energy - bias energy) to the
    trajectory"""

    if isinstance(bias, PlumedBias):
        energy = calculator.calc.get_potential_energy(ase_atoms)

    else:
        energy = calculator.get_potential_energy(ase_atoms)

    energies.append(energy)
    return None


def _save_trajectory(ase_traj, traj_name, **kwargs) -> None:
    """Saves the trajectory with a unique name based on the current simulation
    time"""

    # Prevents initial trajectory save at time == 0
    if len(ase_traj) == 1:
        return None

    specified_key = None
    for key in ['save_ns', 'save_fs', 'save_ps']:
        if key in kwargs:
            specified_key = key
            break

    traj_basename = traj_name[:-5]
    time_units = specified_key.split('_')[-1]
    saving_interval = kwargs[specified_key]

    time = saving_interval
    while os.path.exists(f'{traj_basename}_{time}{time_units}.traj'):
        time += saving_interval

    shutil.copyfile(src=traj_name,
                    dst=f'{traj_basename}_{time}{time_units}.traj')

    return None


def _get_traj_name(restart_files: Optional[List[str]] = None,
                   **kwargs
                   ) -> str:
    """Returns the name of the trajectory which is going to be created
    (or on to which the new frames will be appended in the case of restart)"""

    if restart_files is None:
        if 'idx' in kwargs:
            traj_name = f'trajectory_{kwargs["idx"]}.traj'
        else:
            traj_name = f'trajectory.traj'

        return traj_name

    else:
        for filename in restart_files:
            if filename.endswith('.traj'):
                traj_name = filename

                return traj_name


def _convert_ase_traj(traj_name, bias, **kwargs) -> 'mlptrain.Trajectory':
    """Convert an ASE trajectory into an mlptrain Trajectory"""

    ase_traj = ASETrajectory(traj_name, 'r')
    mlt_traj = Trajectory()

    # Iterate through each frame (set of atoms) in the trajectory
    for atoms in ase_traj:
        config = Configuration()
        config.atoms = [ade.Atom(label) for label in atoms.symbols]

        cell = atoms.cell[:]
        config.box = Box([cell[0][0], cell[1][1], cell[2][2]])

        # Set the atom_pair_list of every atom in the configuration
        for i, position in enumerate(atoms.get_positions()):
            config.atoms[i].coord = position

        mlt_traj.append(config)

    if isinstance(bias, PlumedBias) and bias.setup is None:
        _attach_plumed_coordinates(mlt_traj, bias, **kwargs)

    return mlt_traj


def _attach_plumed_coordinates(mlt_traj, bias, **kwargs) -> None:
    """Attaches PLUMED collective variable values to configurations in the
    trajectory if all colvar files have been printed"""

    colvar_filenames = [_colvar_filename(cv, kwargs) for cv in bias.cvs]

    if all(os.path.exists(fname) for fname in colvar_filenames):

        all_cvs_coordinates = np.zeros((len(mlt_traj), bias.n_cvs))
        for j, fname in enumerate(colvar_filenames):
            all_cvs_coordinates[:, j] = np.loadtxt(fname, usecols=1)

        for i, config in enumerate(mlt_traj):
            config.plumed_coordinates = all_cvs_coordinates[i, :]

    return None


def _set_momenta_and_geometry(ase_atoms:      'ase.atoms.Atoms',
                              temp:           float,
                              bbond_energy:   dict,
                              fbond_energy:   dict,
                              restart_files:  List[str],
                              traj_name:      str
                              ) -> None:
    """Set the initial momenta and geometry of the starting configuration"""

    if restart_files is None:

        if temp > 0:
            logger.info(f'Initialising initial velocities for {temp} K')

            MaxwellBoltzmannDistribution(ase_atoms, temperature_K=temp,
                                         rng=RandomState())
        else:
            # Set the momenta to zero
            ase_atoms.arrays['momenta'] = np.zeros((len(ase_atoms), 3))

        def add_momenta(idx, vector, energy):
            masses = ase_atoms.get_masses()
            ase_atoms.arrays['momenta'][idx] = (np.sqrt(masses[idx] * energy)
                                                * vector)
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

    else:
        logger.info('Initialising starting geometry and momenta from the '
                    'last configuration')

        last_configuration = read(traj_name)

        ase_atoms.set_positions(last_configuration.get_positions())
        ase_atoms.set_momenta(last_configuration.get_momenta())

    return None


def _initialise_traj(ase_atoms:      'ase.atoms.Atoms',
                     restart_files:  List[str],
                     traj_name:      str
                     ) -> 'ase.io.trajectory.Trajectory':
    """Initialise ASE trajectory object"""

    if restart_files is None:
        traj = ASETrajectory(traj_name, 'w', ase_atoms)

    else:
        # Remove the last frame to avoid duplicate frames
        previous_traj = ASETrajectory(traj_name, 'r', ase_atoms)
        previous_atoms = previous_traj[:-1]

        os.remove(traj_name)

        traj = ASETrajectory(traj_name, 'w', ase_atoms)
        for atoms in previous_atoms:
            traj.write(atoms)

    return traj


def _n_simulation_steps(dt: float,
                        kwargs: dict
                        ) -> int:
    """Calculate the number of simulation steps from a set of keyword
    arguments e.g. kwargs = {'fs': 100}

    ---------------------------------------------------------------------------
    Arguments:
        dt: Timestep in fs

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


def _traj_saving_interval(dt: float,
                          kwargs: dict
                          ) -> int:
    """Calculate the interval at which a trajectory is saved"""

    if 'save_ps' in kwargs:
        time_fs = 1E3 * kwargs['save_ps']

    elif 'save_fs' in kwargs:
        time_fs = kwargs['save_fs']

    elif 'save_ns' in kwargs:
        time_fs = 1E6 * kwargs['save_ns']

    else:
        raise ValueError('Saving time not found')

    saving_interval = max(int(time_fs / dt), 1)

    return saving_interval


def _plumed_setup(bias, temp, interval, **kwargs) -> List:
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

    # Defining DOFs and CVs (including upper and lower walls)
    for cv in bias.cvs:
        setup.extend(cv.setup)

    # Metadynamics
    if bias.metadynamics:

        hills_filename = _hills_filename(kwargs)

        if 'load_metad_bias' in kwargs and kwargs['load_metad_bias'] is True:
            load_metad_bias_setup = 'RESTART=YES '

        else:
            load_metad_bias_setup = ''

        metad_setup = ['metad: METAD '
                       f'ARG={bias.cv_sequence} '
                       f'PACE={bias.pace} '
                       f'HEIGHT={bias.height} '
                       f'SIGMA={bias.width_sequence} '
                       f'TEMP={temp} '
                       f'{bias.biasfactor_setup}'
                       f'{bias.metad_grid_setup}'
                       f'{load_metad_bias_setup}'
                       f'FILE={hills_filename}']
        setup.extend(metad_setup)

    # Printing trajectory in terms of DOFs and CVs
    for cv in bias.cvs:

        if cv.dof_names is not None:
            args = f'{cv.name},{cv.dof_sequence}'

        else:
            args = cv.name

        print_setup = ['PRINT '
                       f'ARG={args} '
                       f'FILE={_colvar_filename(cv, kwargs)} '
                       f'STRIDE={interval}']
        setup.extend(print_setup)

    if 'remove_print' in kwargs and kwargs['remove_print'] is True:
        for line in setup:
            if line.startswith('PRINT'):
                setup.remove(line)

    if 'write_plumed_setup' in kwargs and kwargs['write_plumed_setup'] is True:
        with open('plumed_setup.dat', 'w') as f:
            for line in setup:
                f.write(f'{line}\n')

    return setup


def _colvar_filename(cv, kwargs) -> str:
    """Return the name of the file where the trajectory in terms of collective
    variable values will be written"""

    # Remove the dot if component CV is used
    name_without_dot = '_'.join(cv.name.split('.'))

    if 'idx' in kwargs:
        colvar_filename = f'colvar_{name_without_dot}_{kwargs["idx"]}.dat'

    else:
        colvar_filename = f'colvar_{name_without_dot}.dat'

    return colvar_filename


def _hills_filename(kwargs) -> str:
    """Return the name of the file where a list of deposited gaussians will be
    written"""

    filename = 'HILLS'

    if 'iteration' in kwargs and kwargs['iteration'] is not None:
        filename += f'_{kwargs["iteration"]}'

    if 'idx' in kwargs and kwargs['idx'] is not None:
        filename += f'_{kwargs["idx"]}'

    filename += '.dat'
    return filename
