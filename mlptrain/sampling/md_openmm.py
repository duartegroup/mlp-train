import os
from copy import deepcopy
from typing import List, Optional, Sequence, Union

import ase

import mlptrain as mlt
from mlptrain.log import logger
from mlptrain.utils import work_in_tmp_dir
from mlptrain.sampling.md import (
    _convert_ase_traj,
    _get_traj_name,
    _initialise_traj,
    _n_simulation_steps,
    _save_trajectory,
    _traj_saving_interval,
)

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    _HAS_OPENMM = True
except ImportError:
    _HAS_OPENMM = False

try:
    from openmmml import MLPotential

    _HAS_OPENMM_ML = True
except ImportError:
    _HAS_OPENMM_ML = False

# Conversion factor from kJ/mol to eV
_KJ_PER_MOL_TO_EV = (ase.units.kJ / ase.units.mol) / ase.units.eV


def run_mlp_md_openmm(
    configuration: 'mlt.Configuration',
    mlp: 'mlt.potentials._base.MLPotential',
    temp: float,
    dt: float,
    interval: int,
    init_temp: Optional[float] = None,
    fbond_energy: Optional[dict] = None,
    bbond_energy: Optional[dict] = None,
    bias: Optional[Union['mlt.Bias', 'mlt.PlumedBias']] = None,
    restart_files: Optional[List[str]] = None,
    copied_substrings: Optional[Sequence[str]] = None,
    kept_substrings: Optional[Sequence[str]] = None,
    platform: Optional[str] = None,
    **kwargs,
) -> 'mlt.Trajectory':
    """
    Run molecular dynamics on a system using a MLP to predict energies and
    forces and OpenMM to drive dynamics. The function is executed in a temporary
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

        bias: (mlt.Bias | mltr.PlumedBias) mlp-train constrain
              to use in the dynamics

        restart_files: List of files which are needed for restarting the
                       simulation, e.g. 'simulation.state.xml', 'trajectory.traj'

        kept_substrings: List of substrings with which files are copied back
                         from the temporary directory
                         e.g. '.json', 'trajectory_1.traj'

        copied_substrings: List of substrings with which files are copied
                           to the temporary directory. Files required for MLPs
                           are added to the list automatically

        platform: (str) OpenMM platform to use. If None, the fastest available
                    platform is used in this order: 'CUDA', 'OpenCL', 'CPU', 'Reference'.
    ---------------
    Keyword Arguments:

        {fs, ps, ns}: Simulation time in some units

        {save_fs, save_ps, save_ns}: Trajectory saving interval in some units

        constraints: (List) List of ASE constraints to use in the dynamics
                            e.g. [ase.constraints.Hookean(a1, a2, k, rt)]

        write_plumed_setup: (bool) If True saves the PLUMED input file as
                            plumed_setup.dat

    Returns:

        (mlt.Trajectory):
    """
    if (
        fbond_energy is not None
        or bbond_energy is not None
        or bias is not None
    ):
        # TODO: Implement bias and other types of constraints.
        raise NotImplementedError(
            'fbond_energy, bbond_energy and bias arguments not supported yet'
        )

    restart = restart_files is not None

    if copied_substrings is None:
        copied_substrings = []
    if kept_substrings is None:
        kept_substrings = []

    copied_substrings_list = list(copied_substrings)
    kept_substrings_list = list(kept_substrings)

    copied_substrings_list.extend(['.xml', '.json', '.pth', '.model'])

    if restart:
        logger.info('Restarting MLP OpenMM MD')

        if not isinstance(restart_files, list):
            raise TypeError('Restart files must be a list')

        for file in restart_files:
            if not isinstance(file, str):
                raise TypeError(
                    'Restart files must be a list of strings '
                    'specifying filenames'
                )

        if not any(file.endswith('.state.xml') for file in restart_files):
            raise ValueError(
                'Restaring an OpenMM simulation requires a .state.xml file '
                'from the previous simulation'
            )

        if not any(file.endswith('.traj') for file in restart_files):
            raise ValueError(
                'Restaring an OpenMM simulation requires a .traj file '
                'from the previous simulation'
            )

        copied_substrings_list.extend(restart_files)
        kept_substrings_list.extend(restart_files)
    else:
        logger.info('Running MLP MD with OpenMM')

    decorator = work_in_tmp_dir(
        copied_substrings=copied_substrings_list,
        kept_substrings=kept_substrings_list,
    )

    _run_mlp_md_decorated = decorator(_run_mlp_md_openmm)

    traj = _run_mlp_md_decorated(
        configuration=configuration,
        mlp=mlp,
        temp=temp,
        dt=dt,
        interval=interval,
        init_temp=init_temp,
        fbond_energy=fbond_energy,
        bbond_energy=bbond_energy,
        bias=bias,
        restart_files=restart_files,
        platform=platform,
        **kwargs,
    )
    return traj


def _run_mlp_md_openmm(
    configuration: 'mlt.Configuration',
    mlp: 'mlt.potentials._base.MLPotential',
    temp: float,
    dt: float,
    interval: int,
    init_temp: Optional[float] = None,
    fbond_energy: Optional[dict] = None,
    bbond_energy: Optional[dict] = None,
    bias: Optional[Union['mlt.Bias', 'mlt.PlumedBias']] = None,
    restart_files: Optional[List[str]] = None,
    platform: Optional[str] = None,
    **kwargs,
) -> 'mlt.Trajectory':
    """
    Run molecular dynamics on a system using a MLP to predict energies and
    forces and OpenMM to drive dynamics
    """
    restart = restart_files is not None

    if not _HAS_OPENMM:
        raise ImportError(
            'OpenMM is not installed. Install it with '
            "'conda install -c conda-forge openmm'"
        )

    if not _HAS_OPENMM_ML:
        raise ImportError(
            'openmm-ml is not installed. Install it with '
            "'conda install -c conda-forge openmm-ml'"
        )

    # Calculate the number of steps to perform.
    n_steps = _n_simulation_steps(dt=dt, kwargs=kwargs)

    # Set the box size if required
    if mlp.requires_non_zero_box_size and configuration.box is None:
        logger.warning('Assuming vacuum simulation. Box size = 1000 nm^3')
        configuration.box = mlt.Box([100, 100, 100])

    # Get the ASE atoms object and positions.
    ase_atoms = configuration.ase_atoms

    # Get the name of the trajectory and simulation state files.
    traj_name = _get_traj_name(restart_files=restart_files, **kwargs)
    simulation_name = _get_simulation_name(
        restart_files=restart_files, **kwargs
    )

    # Create the OpenMM topology
    topology = _create_openmm_topology(ase_atoms)

    # Get the OpenMM platform
    platform = _get_openmm_platform(platform)

    # Create the OpenMM simulation object
    simulation = _create_openmm_simulation(
        mlp=mlp,
        topology=topology,
        temp=temp,
        dt=dt,
        platform=platform,
    )

    # Set the initial positions and velocities
    _set_momenta_and_geometry(
        simulation=simulation,
        positions=ase_atoms.get_positions() * unit.angstrom,
        temp=init_temp if init_temp is not None else temp,
        restart_file=simulation_name if restart else None,
    )

    # Initialise the ASE trajectory with the last frame of the previous trajectory
    ase_traj = _initialise_traj(
        ase_atoms=ase_atoms,
        restart=restart,
        traj_name=traj_name,
        remove_last=False,
    )

    # If MD is restarted, energies of frames from the previous trajectory
    # are not loaded. Setting them to None
    energies = [None for _ in range(len(ase_traj))]
    biased_energies = deepcopy(energies)
    bias_energies = deepcopy(energies)

    # Calculate the number of steps already performed.
    n_previous_steps = interval * len(ase_traj)

    logger.info(
        f'Running OpenMM simulation for {n_steps} steps with saving interval {interval}'
    )

    # Run the dynamics
    _run_dynamics(
        simulation=simulation,
        simulation_name=simulation_name,
        ase_atoms=ase_atoms,
        ase_traj=ase_traj,
        traj_name=traj_name,
        dt=dt,
        interval=interval,
        n_steps=n_steps,
        n_previous_steps=n_previous_steps,
        energies=energies,
        biased_energies=biased_energies,
        **kwargs,
    )

    # Close the ASE trajectory
    ase_traj.close()

    # Duplicate frames removed only if PLUMED bias is initialised not from file
    # if restart and isinstance(bias, PlumedBias) and not bias.from_file:
    #    _remove_colvar_duplicate_frames(bias=bias, **kwargs)

    traj = _convert_ase_traj(traj_name=traj_name, bias=bias, **kwargs)

    for energy, biased_energy in zip(energies, biased_energies):
        if energy is not None and biased_energy is not None:
            bias_energy = biased_energy - energy
            bias_energies.append(bias_energy)

    for i, (frame, energy, bias_energy) in enumerate(
        zip(traj, energies, bias_energies)
    ):
        frame.update_attr_from(configuration)
        frame.energy.predicted = energy
        frame.energy.bias = bias_energy
        frame.time = dt * interval * i

    return traj


# ============================================================================= #
#             Auxiliary functions to create the OpenMM Simulation               #
# ============================================================================= #
def _create_openmm_topology(ase_atoms: 'ase.Atoms') -> 'app.Topology':
    """Create an OpenMM topology from an ASE atoms object."""
    logger.info('Creating the OpenMM topology')
    topology = app.Topology()
    chain = topology.addChain()

    atomic_numbers = ase_atoms.get_atomic_numbers()

    for atomic_number in atomic_numbers:
        residue = topology.addResidue(name='X', chain=chain)
        element = app.Element.getByAtomicNumber(atomic_number)
        topology.addAtom(element.name, element, residue)

    return topology


def _get_openmm_platform(platform: Optional[str]) -> 'mm.Platform':
    """Get the OpenMM platform to use."""
    import torch

    available_platforms = [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ]

    # OpenMM might have been built with CUDA support
    # but the current system might not have a GPU available (typical in clusters)
    if 'CUDA' in available_platforms and not torch.cuda.is_available():
        available_platforms.remove('CUDA')

    if platform is not None and platform in available_platforms:
        platform = mm.Platform.getPlatformByName(platform)
    else:
        platform = next(
            (
                p
                for p in ['CUDA', 'OpenCL', 'CPU', 'Reference']
                if p in available_platforms
            ),
            None,
        )
        if platform is None:
            raise ValueError(
                f'No suitable platform found. Available platforms are: {available_platforms}'
            )
        platform = mm.Platform.getPlatformByName(platform)

    logger.info(f'Using the OpenMM platform: {platform.getName()}')

    return platform


def _create_openmm_simulation(
    mlp: 'mlt.potentials._base.MLPotential',
    topology: 'app.Topology',
    temp: float,
    dt: float,
    platform: 'mm.Platform',
) -> 'app.Simulation':
    """Create an OpenMM simulation object."""
    logger.info('Creating the OpenMM simulation object')

    # Use the mace model with openmm-ml and make sure the total energy is used.
    potential = MLPotential('mace', modelPath=mlp.filename)
    system = potential.createSystem(topology, returnEnergyType='energy')

    # Use a Langevin integrator if temp>0 (NVT ensemble).
    # Otherwise, use a Verlet integrator (NVE ensemble).
    if temp > 0:
        logger.info(
            f'Using Langevin integrator (NVT) with temperture={temp} K'
        )
        integrator = mm.LangevinMiddleIntegrator(
            temp * unit.kelvin, 1.0 / unit.picoseconds, dt * unit.femtoseconds
        )
    else:
        logger.info(f'Using Verlet integrator (NVE) as temperture is {temp} K')
        integrator = mm.VerletIntegrator(dt * unit.femtoseconds)

    simulation = app.Simulation(topology, system, integrator, platform)

    return simulation


def _set_momenta_and_geometry(
    simulation: 'app.Simulation',
    positions: 'unit.Quantity',
    temp: float,
    restart_file: Optional[str] = None,
) -> 'app.Simulation':
    """Set the momenta and geometry for the OpenMM simulation."""

    if restart_file is not None:
        if os.path.isfile(restart_file):
            logger.info(
                f'Restarting the OpenMM simulation state from file {restart_file}'
            )
            simulation.loadState(restart_file)
        else:
            raise FileNotFoundError(f'File {restart_file} not found')
    else:
        logger.info(
            'Setting the initial momenta and geometry for the OpenMM simulation'
        )
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temp * unit.kelvin)

    return simulation


def _get_simulation_name(
    restart_files: Optional[List[str]] = None, **kwargs
) -> str:
    """Return the name of the OpenMM simulation to be created or restarted."""
    if restart_files is None:
        if 'idx' in kwargs:
            simulation_name = f'simulation_{kwargs["idx"]}.state.xml'
        else:
            simulation_name = 'simulation.state.xml'

        return simulation_name
    else:
        for filename in restart_files:
            if filename.endswith('.state.xml'):
                return filename

    raise FileNotFoundError(
        'Restart mode detected, but no simulation state files were found in restart_files. '
    )


# ============================================================================= #
#               Auxiliary functions to run the OpenMM Simulation                #
# ============================================================================= #
def _run_dynamics(
    simulation: 'app.Simulation',
    simulation_name: str,
    ase_atoms: 'ase.Atoms',
    ase_traj: 'ase.io.trajectory.Trajectory',
    traj_name: str,
    dt: float,
    interval: int,
    n_steps: int,
    n_previous_steps: int,
    energies: List[Optional[float]],
    biased_energies: List[Optional[float]],
    **kwargs,
) -> None:
    """Run the MD and save frames to the mlt.Trajectory."""

    def append_unbiased_energy():
        """Append the unbiased potential energy to the energies list."""
        energies.append(potential_energy)

    def append_biased_energy():
        """Append the biased potential energy to the biased_energies list."""
        biased_energies.append(biased_energy)

    def save_trajectory():
        """Save the ASE trajectory to a file."""
        _save_trajectory(ase_traj, traj_name, **kwargs)

    def save_simulation_state():
        """Save the state of the OpenMM simulation."""
        simulation.saveState(simulation_name)

    def _add_frame_to_ase_traj():
        """Add a new frame to the ASE train trajectory"""
        # Create a new ASE atoms object.
        new_ase_atoms = ase.Atoms(
            symbols=ase_atoms.get_chemical_symbols(),
            positions=coordinates,
            cell=ase_atoms.get_cell(),
        )

        # Append the new frame to the trajectory.
        ase_traj.write(new_ase_atoms, energy=potential_energy)

    # Determine saving intervals
    if any(key in kwargs for key in ['save_fs', 'save_ps', 'save_ns']):
        traj_saving_interval = _traj_saving_interval(kwargs)
    else:
        traj_saving_interval = 0

    # Run the dynamics n_steps, performing interval steps at a time.

    for j in range(n_previous_steps // interval, n_steps // interval):
        logger.info(f'Step {j + 1} / {n_steps // interval}')
        simulation.step(interval)
        time = dt * interval * (j + 1)

        # Get the coordinates and energy of the system from the OpenMM simulation.
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        coordinates = state.getPositions(asNumpy=True).value_in_unit(
            unit.angstrom
        )
        potential_energy = (
            state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            * _KJ_PER_MOL_TO_EV
        )

        # TODO: Implement biased_energy when bias is implemented
        biased_energy = potential_energy

        # Add the frame to the ASE trajectory.
        _add_frame_to_ase_traj()

        # Store the energies
        append_unbiased_energy()
        append_biased_energy()
        save_simulation_state()

        if traj_saving_interval > 0 and time % traj_saving_interval == 0:
            save_trajectory()
