import copy
from typing import Optional, Sequence, List

import mlptrain as mlt


def run_mlp_md_openmm(configuration: 'mlt.Configuration',
               mlp:                'mlt.potentials._base.MLPotential',
               temp:               float,
               dt:                 float,
               interval:           int,
               init_temp:          Optional[float] = None,
               fbond_energy:       Optional[dict] = None,
               bbond_energy:       Optional[dict] = None,
               bias:               Optional = None,
               restart_files:      Optional[List[str]] = None,
               copied_substrings:  Optional[Sequence[str]] = None,
               kept_substrings:    Optional[Sequence[str]] = None,
               **kwargs
               ) -> 'mlt.Trajectory':
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

        bias: (mlt.Bias | mltr.PlumedBias) mlp-train constrain
              to use in the dynamics

        restart_files: List of files which are needed for restarting the
                       simulation

        kept_substrings: List of substrings with which files are copied back
                         from the temporary directory
                         e.g. '.json', 'trajectory_1.traj'

        copied_substrings: List of substrings with which files are copied
                           to the temporary directory. Files required for MLPs
                           are added to the list automatically
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


    restart = restart_files is not None

    if copied_substrings is None:
        copied_substrings = []
    if kept_substrings is None:
        kept_substrings = []

    copied_substrings_list = list(copied_substrings)
    kept_substrings_list = list(kept_substrings)

    copied_substrings_list.extend(['.xml', '.json', '.pth', '.model'])

    if restart:
        mlt.log.logger.error('Restarting MLP MD with OpenMM not implemented')
        mlt.log.logger.info('Restarting MLP MD')

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
        mlt.log.logger.info('Running MLP MD')



    traj_openmm = _run_mlp_md_openmm(configuration=configuration,
                                 mlp=mlp,
                                 temp=temp,
                                 dt=dt,
                                 interval=interval,
                                 init_temp=init_temp,
                                 fbond_energy=fbond_energy,
                                 bbond_energy=bbond_energy,
                                 bias=bias,
                                 restart_files=restart_files,
                                 **kwargs)


    return traj_openmm



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
        mlt.log.logger.warning('Unexpectedly small or large timestep - is it in fs?')

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


def _run_mlp_md_openmm(configuration:  'mlt.Configuration',
                mlp:            'mlt.potentials._base.MLPotential',
                temp:           float,
                dt:             float,
                interval:       int,
                init_temp:      Optional[float] = None,
                fbond_energy:   Optional[dict] = None,
                bbond_energy:   Optional[dict] = None,
                bias:           Optional = None,
                restart_files:  Optional[List[str]] = None,
                **kwargs
                ) -> 'mlt.Trajectory':
    
    """
    Run molecular dynamics on a system using a MLP to predict energies and
    forces and OpenMM to drive dynamics
    """

    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError:
        raise ImportError("Cannot import OpenMM. Please make sure OpenMM is installed.")
    
    try:
        from openmmml import MLPotential
    except ImportError:
        raise ImportError("Cannot import OpenMM-ML")
    
    from sys import stdout
    
    n_steps = _n_simulation_steps(dt=dt, kwargs=kwargs)

    if mlp.requires_non_zero_box_size and configuration.box is None:
        mlt.log.logger.warning('Assuming vaccum simulation. Box size = 1000 nm^3')
        configuration.box = mlt.Box([100, 100, 100])


    ase_atoms = configuration.ase_atoms

    # create OpenMM topology
    topology = app.Topology()
    chain = topology.addChain()

    positions = ase_atoms.get_positions()*unit.angstrom
    atomic_numbers = ase_atoms.get_atomic_numbers()

    #print(positions, atomic_numbers)

    for atomic_number in atomic_numbers:
        residue = topology.addResidue(name='X', chain=chain)
        element = app.Element.getByAtomicNumber(atomic_number)
        topology.addAtom(element.name,element,residue)

    #print(topology)

    # use the mace model with openmm-ml
    # make sure total energy is used
    potential = MLPotential('mace', model_path=mlp.filename)
    system = potential.createSystem(topology, interaction_energy=False)

    # setup OpenMM simulation with Langevin dynamics
    integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, dt*unit.femtoseconds)
    simulation=app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temp*unit.kelvin)

    interval = int(interval)

    simulation.reporters.append(app.StateDataReporter(stdout, interval, step=True,
            potentialEnergy=True, temperature=True, speed=True))


    # create MLP train trajectory to save frames into

    mlt_traj = mlt.Trajectory()

    
    print("running using OpenMM for ", n_steps, " steps with saving interval", interval)

    # add the first config using energies from current MLP
    state = simulation.context.getState(getPositions=True, getEnergy=True)

    coordinates = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    config = mlt.Configuration()
    config.atoms = copy.deepcopy(configuration.atoms)
    config.box = copy.deepcopy(configuration.box)

    for i, position in enumerate(coordinates):
        config.atoms[i].coord = position

    config.energy.predicted = energy/96.48530749925793 # kj/mol -> eV
    config.time = 0.0

    mlt_traj.append(config)

    # now run for n_steps saving at every interval
    for j in range (n_steps//interval):
        simulation.step(interval)

        state = simulation.context.getState(getPositions=True, getEnergy=True)

        coordinates = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        config = mlt.Configuration()
        config.atoms = copy.deepcopy(configuration.atoms)
        config.box = copy.deepcopy(configuration.box)

        for i, position in enumerate(coordinates):
            config.atoms[i].coord = position

        config.energy.predicted = energy/96.48530749925793 # kJ/mol -> eV
        config.time = dt*interval*(j+1)

        mlt_traj.append(config)

    return mlt_traj

