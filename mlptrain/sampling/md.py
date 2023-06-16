import os
import shutil
import numpy as np
import autode as ade
from copy import deepcopy
from typing import Optional, Sequence, List
from numpy.random import RandomState
from mlptrain.configurations import Configuration, Trajectory
from mlptrain.config import Config
from mlptrain.sampling.plumed import (
    PlumedBias,
    PlumedCalculator,
    plumed_setup,
    get_colvar_filename
)
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
               copied_substrings:  Sequence[str] = ('.xml', '.json', '.pth', '.model'),
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

        bias: (mlptrain.Bias | mlptrain.PlumedBias) mlp-train constrain
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
                            e.g. [ase.constraints.Hookean(a1, a2, k, rt)]

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

        bias: (mlptrain.Bias | mlptrain.PlumedBias) mlp-train constrain
              to use in the dynamics

        restart_files: List of files which are needed for restarting the
                       simulation
    ---------------
    Keyword Arguments:

        {fs, ps, ns}: Simulation time in some units

        {save_fs, save_ps, save_ns}: Trajectory saving interval in some units

        constraints: (List) List of ASE constraints to use in the dynamics
                            e.g. [ase.constraints.Hookean(a1, a2, k, rt)]

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
                              restart=restart,
                              traj_name=traj_name)

    ase_traj = _initialise_traj(ase_atoms, restart, traj_name)

    # If MD is restarted, energies of frames from the previous trajectory
    # are not loaded. Setting them to None
    energies = [None for _ in range(len(ase_traj))]
    biased_energies = deepcopy(energies)
    bias_energies = deepcopy(energies)

    n_previous_steps = interval * len(ase_traj)
    _attach_calculator_and_constraints(ase_atoms, mlp, bias, temp,
                                       interval, dt_ase, restart,
                                       n_previous_steps, **kwargs)

    _run_dynamics(ase_atoms, ase_traj, traj_name, interval, temp, dt, dt_ase,
                  n_steps, energies, biased_energies, **kwargs)

    # Duplicate frames removed only if PLUMED bias is initialised not from file
    if restart and isinstance(bias, PlumedBias) and not bias.from_file:
        _remove_colvar_duplicate_frames(bias, **kwargs)

    traj = _convert_ase_traj(traj_name, bias, **kwargs)

    for energy, biased_energy in zip(energies, biased_energies):
        if energy is not None and biased_energy is not None:
            bias_energy = biased_energy - energy
            bias_energies.append(bias_energy)

    for i, (frame, energy, bias_energy) in enumerate(zip(traj, energies, bias_energies)):
        frame.update_attr_from(configuration)
        frame.energy.predicted = energy
        frame.energy.bias = bias_energy
        frame.time = dt * interval * i

    return traj


def _attach_calculator_and_constraints(ase_atoms, mlp, bias, temp, interval,
                                       dt_ase, restart, n_previous_steps,
                                       **kwargs) -> None:
    """Set up the calculator and attach it to the ase_atoms together with bias
    and constraints"""

    if isinstance(bias, PlumedBias):
        logger.info('Using PLUMED bias for MLP MD')

        setup = plumed_setup(bias, temp, interval, **kwargs)
        bias.write_cv_files()

        plumed_calc = PlumedCalculator(calc=mlp.ase_calculator,
                                       input=setup,
                                       timestep=dt_ase,
                                       atoms=ase_atoms,
                                       kT=temp*ase_units.kB,
                                       restart=restart)

        if restart:
            plumed_calc.istep = n_previous_steps

        ase_atoms.calc = plumed_calc

    else:
        ase_atoms.calc = mlp.ase_calculator

    if 'constraints' in kwargs and kwargs['constraints'] is not None:
        constraints = deepcopy(kwargs['constraints'])
    else:
        constraints = []

    if bias is not None:
        constraints.append(bias)

    ase_atoms.set_constraint(constraints)

    return None


def _run_dynamics(ase_atoms, ase_traj, traj_name, interval, temp, dt, dt_ase,
                  n_steps, energies, biased_energies, **kwargs) -> None:
    """Initialise dynamics object and run dynamics"""

    if temp > 0:                                        # Default Langevin NVT
        dyn = Langevin(ase_atoms, dt_ase,
                       temperature_K=temp,
                       friction=0.02)
    else:                                               # Otherwise NVE
        dyn = VelocityVerlet(ase_atoms, dt_ase)

    def append_unbiased_energy():
        energies.append(ase_atoms.calc.get_potential_energy(ase_atoms))

    def append_biased_energy():
        biased_energies.append(ase_atoms.get_potential_energy())

    def save_trajectory():
        _save_trajectory(ase_traj, traj_name, **kwargs)

    dyn.attach(append_unbiased_energy, interval=interval)
    dyn.attach(append_biased_energy, interval=interval)
    dyn.attach(ase_traj.write, interval=interval)

    if any(key in kwargs for key in ['save_fs', 'save_ps', 'save_ns']):
        dyn.attach(save_trajectory,
                   interval=_traj_saving_interval(dt, kwargs))

    logger.info(f'Running {n_steps:.0f} steps with a timestep of {dt} fs')
    dyn.run(steps=n_steps)

    if isinstance(ase_atoms.calc, PlumedCalculator):
        # The calling process waits until PLUMED process has finished
        ase_atoms.calc.plumed.finalize()

    return None


def _save_trajectory(ase_traj, traj_name, **kwargs) -> None:
    """Save the trajectory with a unique name based on the current simulation
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
    """Return the name of the trajectory which is going to be created
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

    if isinstance(bias, PlumedBias) and not bias.from_file:
        _attach_plumed_coordinates(mlt_traj, bias, **kwargs)

    return mlt_traj


def _attach_plumed_coordinates(mlt_traj, bias, **kwargs) -> None:
    """Attach PLUMED collective variable values to configurations in the
    trajectory if all colvar files have been printed"""

    colvar_filenames = [get_colvar_filename(cv, **kwargs) for cv in bias.cvs]

    if all(os.path.exists(fname) for fname in colvar_filenames):

        for config in mlt_traj:
            config.plumed_coordinates = np.zeros(bias.n_cvs)

        for i, cv in enumerate(bias.cvs):
            colvar_fname = colvar_filenames[i]
            cv_values = np.loadtxt(colvar_fname, usecols=1)

            for j, config in enumerate(mlt_traj):
                config.plumed_coordinates[i] = cv_values[j]

    return None


def _set_momenta_and_geometry(ase_atoms:      'ase.atoms.Atoms',
                              temp:           float,
                              bbond_energy:   dict,
                              fbond_energy:   dict,
                              restart:        bool,
                              traj_name:      str
                              ) -> None:
    """Set the initial momenta and geometry of the starting configuration"""

    if not restart:

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
                     restart:        bool,
                     traj_name:      str
                     ) -> 'ase.io.trajectory.Trajectory':
    """Initialise ASE trajectory object"""

    if not restart:
        traj = ASETrajectory(traj_name, 'w', ase_atoms)

    else:
        # Remove the last frame to avoid duplicate frames
        previous_atoms = read(traj_name, index=':-1')
        os.remove(traj_name)

        traj = ASETrajectory(traj_name, 'w', ase_atoms)
        for atoms in previous_atoms:
            traj.write(atoms)

        logger.info('Trajectory has been loaded')

    return traj


def _n_simulation_steps(dt: float,
                        kwargs: dict
                        ) -> int:
    """
    Calculate the number of simulation steps from a set of keyword
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


def _remove_colvar_duplicate_frames(bias, **kwargs) -> None:
    """Remove duplicate frames from generated colvar files when using PLUMED
    bias"""

    colvar_filenames = [get_colvar_filename(cv, **kwargs) for cv in bias.cvs]

    for filename in colvar_filenames:

        with open(filename, 'r') as f:
            lines = f.readlines()

        duplicate_index = None
        for i, line in enumerate(lines):
            if line.startswith('#!') and i != 0:

                # First frame before redundant header is a duplicate
                duplicate_index = i - 1
                break

        if duplicate_index is None:
            raise TypeError(f'Duplicate frame in {filename} was not found')

        lines.pop(duplicate_index)

        with open(filename, 'w') as f:
            for line in lines:
                f.write(line)

    return None
