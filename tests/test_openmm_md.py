import os
import numpy as np
import mlptrain as mlt
import pytest
from ase.io.trajectory import Trajectory as ASETrajectory

from .data.utils import work_in_zipped_dir

import ase.units

here = os.path.abspath(os.path.dirname(__file__))

# All tests should have 'test_openmm' in their name so that they are skipped for the GAP CI run.


@pytest.fixture
def h2_system_config(h2):
    system = mlt.System(h2, box=[50, 50, 50])
    config = system.random_configuration()
    return system, config


@pytest.fixture
def h2o_system_config(h2o):
    system = mlt.System(h2o, box=[50, 50, 50])
    config = system.random_configuration()
    return system, config


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_openmm_topology(h2_system_config, h2o_system_config):
    """Test the creation of an OpenMM Topology from an ASE Atoms object."""
    # H2 molecule
    _, config = h2_system_config
    atoms = config.ase_atoms
    topology = mlt.md_openmm._create_openmm_topology(atoms)

    assert topology.getNumAtoms() == len(atoms)
    assert topology.getNumResidues() == len(atoms)
    assert topology.getNumChains() == 1
    assert topology.getNumBonds() == 0
    assert np.allclose(
        topology.getPeriodicBoxVectors()._value * 10.0, atoms.get_cell().array
    )

    # H2O molecule
    _, config = h2o_system_config
    atoms = config.ase_atoms
    topology = mlt.md_openmm._create_openmm_topology(atoms)

    assert topology.getNumAtoms() == len(atoms)
    assert topology.getNumResidues() == len(atoms)
    assert topology.getNumChains() == 1
    assert topology.getNumBonds() == 0
    assert np.allclose(
        topology.getPeriodicBoxVectors()._value * 10.0, atoms.get_cell().array
    )


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_openmm_simulation(h2o_system_config):
    """Test the OpenMM Simulation object."""
    # H2O molecule
    system, config = h2o_system_config
    atoms = config.ase_atoms
    mace = mlt.potentials.MACE('water', system=system)

    topology = mlt.md_openmm._create_openmm_topology(atoms)
    platform = mlt.md_openmm._get_openmm_platform()

    simulation = mlt.md_openmm._create_openmm_simulation(
        mlp=mace,
        topology=topology,
        temp=300,
        dt=1,
        platform=platform,
    )

    assert np.isclose(simulation.integrator.getTemperature()._value, 300)
    assert np.isclose(simulation.integrator.getStepSize()._value, 0.001)

    mlt.md_openmm._set_momenta_and_geometry(
        simulation=simulation,
        positions=atoms.get_positions() * 0.1,
        temp=0.0,
        restart_file=None,
    )

    # Check that potential and kinetic energies are correct
    # and consistent with ASE
    reference_pot_energy = -13310.4853515625  # kJ/mol
    openmm_pot_energy = (
        simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
    )
    atoms.set_calculator(mace.ase_calculator)
    ase_pot_energy = atoms.get_potential_energy() / (
        (ase.units.kJ / ase.units.mol) / ase.units.eV
    )

    assert np.isclose(openmm_pot_energy, reference_pot_energy)
    assert np.isclose(ase_pot_energy, reference_pot_energy)
    assert np.isclose(ase_pot_energy, openmm_pot_energy)


def test_openmm_simulation_name_generation():
    """Test the simulation name generation."""
    name = mlt.md_openmm._get_simulation_name()
    assert name == 'simulation.state.xml'

    name = mlt.md_openmm._get_simulation_name(idx=2)
    assert name == 'simulation_2.state.xml'

    state_file = 'file1.state.xml'
    name = mlt.md_openmm._get_simulation_name(restart_files=[state_file])
    assert name == state_file


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_openmm_md(h2o_system_config):
    """Test the OpenMM MD simulation."""
    # H2O molecule
    system, config = h2o_system_config

    mace = mlt.potentials.MACE('water', system=system)

    # Run some dynamics with the potential
    mlt.md_openmm.run_mlp_md_openmm(
        configuration=config,
        mlp=mace,
        temp=300,
        dt=1,
        interval=10,
        fs=100,
        kept_substrings=['.state.xml', '.traj'],
    )

    traj = ASETrajectory('trajectory.traj')
    assert os.path.exists('simulation.state.xml')
    assert os.path.exists('trajectory.traj')
    # 100 fs simulation with dt = 1 fs and interval of 10 -> 11 frames
    assert len(traj) == 11


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_openmm_md_restart(h2o_system_config):
    """Test the MD restart functionality."""
    system, config = h2o_system_config
    atoms = config.ase_atoms
    mace = mlt.potentials.MACE('water', system=system)
    initial_trajectory = ASETrajectory('md_restart_h2o.traj', 'r', atoms)

    mlt.md_openmm.run_mlp_md_openmm(
        configuration=config,
        mlp=mace,
        temp=300,
        dt=1,
        interval=10,
        restart_files=['md_restart_h2o.traj', 'md_restart_h2o.state.xml'],
        fs=100,
    )

    assert os.path.exists('md_restart_h2o.traj')

    final_trajectory = ASETrajectory('md_restart_h2o.traj', 'r', atoms)

    # 10 ps simulation with dt = 1 fs and interval of 10 -> 1001 frames
    assert len(initial_trajectory) == 1001

    # Adding 1 ps simulation with interval 10 -> 101 frames, but removing one
    # duplicate frame
    assert len(final_trajectory) == 1001 + 11 - 1


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_openmm_md_save(h2o_system_config):
    """Test the MD save functionality."""
    system, config = h2o_system_config
    mace = mlt.potentials.MACE('water', system=system)

    mlt.md_openmm.run_mlp_md_openmm(
        configuration=config,
        mlp=mace,
        temp=300,
        dt=1,
        interval=10,
        kept_substrings=['.traj'],
        fs=100,
        save_fs=20,
    )

    assert os.path.exists('trajectory.traj')
    assert not os.path.exists('trajectory_0fs.traj')
    assert os.path.exists('trajectory_20fs.traj')
    assert os.path.exists('trajectory_100fs.traj')
    assert not os.path.exists('trajectory_120fs.traj')

    traj_20fs = ASETrajectory('trajectory_20fs.traj')

    # 20 fs / 10 interval == 2 frames; + 1 starting frame
    assert len(traj_20fs) == 2 + 1
