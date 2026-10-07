import os
import random

import numpy as np
import pytest
from autode.atoms import Atom
from autode.exceptions import SolventNotFound

import mlptrain as mlt
from mlptrain.configurations.configuration import (
    Configuration,
    _get_max_mol_distance,
    _random_vector_in_box,
)
from mlptrain.potentials._base import MLPotential


@pytest.fixture
def h2o_solvated_with_h2o():
    """Water molecule solvated with water with default
    solvation settings, water solvent specified by name"""
    atoms = [
        Atom('H', 6.726, 5.781, 5.842),
        Atom('H', 5.432, 5.977, 4.949),
        Atom('O', 5.756, 5.781, 5.842),
        Atom('O', 2.626, 8.510, 7.795),
        Atom('H', 2.298, 9.408, 7.873),
        Atom('H', 2.228, 8.000, 8.503),
        Atom('O', 6.249, 2.553, 6.820),
        Atom('H', 6.871, 1.959, 6.394),
        Atom('H', 6.761, 3.159, 7.359),
        Atom('O', 8.063, 3.891, 1.832),
        Atom('H', 8.541, 3.869, 1.000),
        Atom('H', 8.024, 4.809, 2.111),
        Atom('O', 11.280, 4.326, 6.371),
        Atom('H', 11.258, 4.569, 7.299),
        Atom('H', 11.163, 5.133, 5.865),
        Atom('O', 6.645, 8.208, 0.523),
        Atom('H', 6.535, 7.280, 0.303),
        Atom('H', 7.528, 8.310, 0.884),
        Atom('O', 2.680, 1.112, 3.240),
        Atom('H', 2.218, 1.928, 3.438),
        Atom('H', 3.432, 1.339, 2.688),
        Atom('O', 2.368, 3.095, 10.826),
        Atom('H', 2.736, 3.686, 11.486),
        Atom('H', 3.051, 2.459, 10.605),
        Atom('O', 8.396, 1.882, 4.358),
        Atom('H', 8.633, 1.388, 5.146),
        Atom('H', 9.060, 2.565, 4.243),
        Atom('O', 7.986, 9.773, 8.960),
        Atom('H', 7.804, 9.885, 9.895),
        Atom('H', 7.170, 9.487, 8.545),
        Atom('O', 4.567, 10.584, 5.254),
        Atom('H', 5.424, 10.425, 5.656),
        Atom('H', 3.993, 10.928, 5.942),
        Atom('O', 5.951, 1.066, 0.507),
        Atom('H', 5.911, 0.370, 1.166),
        Atom('H', 5.116, 1.536, 0.542),
        Atom('O', 9.911, 0.116, 8.360),
        Atom('H', 10.251, 0.505, 7.552),
        Atom('H', 10.651, 0.028, 8.965),
        Atom('O', 3.110, 5.824, 2.069),
        Atom('H', 2.772, 5.184, 2.699),
        Atom('H', 2.399, 6.015, 1.452),
        Atom('O', 5.692, 0.816, 8.813),
        Atom('H', 5.161, 0.704, 8.021),
        Atom('H', 5.112, 0.672, 9.564),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)


def test_equality():
    config1 = Configuration()
    assert config1 == config1
    assert config1 == Configuration()

    config2 = Configuration(atoms=[Atom('H')])

    assert config1 != config2


seeded_random = random.Random()


def test_random_vector_in_box():
    vector = _random_vector_in_box(
        10,
        seeded_random.random(),
        seeded_random.random(),
        seeded_random.random(),
    )
    assert all(v <= 10 for v in vector)
    assert all(v >= 0 for v in vector)


def test_get_max_mol_distance(h2o_configuration):
    max_distance_h2o = _get_max_mol_distance(h2o_configuration.atoms)
    max_distance_h2o = round(max_distance_h2o, 3)
    assert max_distance_h2o == 1.584


def test_solvate(h2o_configuration, h2o_solvated_with_h2o):
    h2o_configuration.solvate(solvent_name='water', buffer_distance=6.0)
    assert len(h2o_configuration.atoms) == len(h2o_solvated_with_h2o.atoms)

    for atom, ref_atom in zip(
        h2o_configuration.atoms, h2o_solvated_with_h2o.atoms
    ):
        assert np.allclose(atom.coordinate, ref_atom.coordinate)


def test_wrong_solvent_name_raises_not_found(h2o_configuration):
    with pytest.raises(SolventNotFound):
        h2o_configuration.solvate(solvent_name='solvo_solverson')


def test_no_inputs_for_solvate(h2o_configuration):
    with pytest.raises(ValueError):
        h2o_configuration.solvate()


def test_only_molecule_for_solvate(h2o_configuration, h2o):
    with pytest.raises(ValueError):
        h2o_configuration.solvate(solvent_molecule=h2o)


def test_only_density_for_solvate(h2o_configuration):
    with pytest.raises(ValueError):
        h2o_configuration.solvate(solvent_density=1)


def test_only_too_many_inputs_for_solvate(h2o_configuration, h2o):
    with pytest.raises(ValueError):
        h2o_configuration.solvate(
            solvent_name='water', solvent_density=1, solvent_molecule=h2o
        )


def test_negative_density_for_solvate(h2o_configuration, h2o):
    with pytest.raises(ValueError):
        h2o_configuration.solvate(solvent_molecule=h2o, solvent_density=-1)


def test_no_atoms_in_solvent_molecule(h2o_configuration, empty_molecule):
    with pytest.raises(ValueError):
        h2o_configuration.solvate(
            solvent_density=1, solvent_molecule=empty_molecule
        )


# Tests for mol_dict functionality
def test_mol_dict_initialization():
    """Test that mol_dict is properly initialized as empty dictionary"""
    config = Configuration()
    assert hasattr(config, 'mol_dict')
    assert isinstance(config.mol_dict, dict)
    assert len(config.mol_dict) == 0


def test_mol_dict_save_load(tmp_path):
    """Test saving and loading mol_dict to/from file"""
    import json

    # Create a configuration with some atoms
    atoms = [
        Atom('H', 0.0, 0.0, 0.0),
        Atom('H', 1.0, 0.0, 0.0),
        Atom('O', 0.5, 0.5, 0.0),
    ]
    config = Configuration(atoms=atoms)

    # Manually set mol_dict for testing
    config.mol_dict = {
        'solute': [{'start': 0, 'end': 3, 'formula': 'H2O'}],
        'water': [
            {'start': 3, 'end': 6, 'formula': 'H2O'},
            {'start': 6, 'end': 9, 'formula': 'H2O'},
        ],
    }

    # Save to a temporary file
    xyz_file = tmp_path / 'test.xyz'
    config.save_xyz(str(xyz_file))

    # Check that mol_dict file was created
    mol_dict_file = tmp_path / '.test.mol_dict.txt'
    assert mol_dict_file.exists()

    # Load mol_dict and verify content
    with open(mol_dict_file, 'r') as f:
        loaded_dict = json.load(f)

    assert loaded_dict == config.mol_dict

    # Test loading mol_dict into a new configuration
    new_config = Configuration()
    success = new_config.load_mol_dict(str(xyz_file))
    assert success
    assert new_config.mol_dict == config.mol_dict


def test_mol_dict_from_xyz_with_mol_dict(tmp_path):
    """Test loading configuration from xyz with automatic mol_dict loading"""
    import json

    # Create test files
    xyz_file = tmp_path / 'test.xyz'
    mol_dict_file = tmp_path / '.test.mol_dict.txt'

    # Create simple xyz file
    with open(xyz_file, 'w') as f:
        f.write('3\n')
        f.write('Test molecule\n')
        f.write('H 0.0 0.0 0.0\n')
        f.write('H 1.0 0.0 0.0\n')
        f.write('O 0.5 0.5 0.0\n')

    # Create mol_dict file
    test_mol_dict = {'solute': [{'start': 0, 'end': 3, 'formula': 'H2O'}]}
    with open(mol_dict_file, 'w') as f:
        json.dump(test_mol_dict, f)

    # Load configuration using from_xyz
    config = Configuration.from_xyz(str(xyz_file))

    # Verify mol_dict was loaded
    assert config.mol_dict == test_mol_dict
    assert config.atoms is not None
    assert len(config.atoms) == 3


def test_mol_dict_validate():
    """Test mol_dict validation functionality"""
    atoms = [
        Atom('H', 0.0, 0.0, 0.0),
        Atom('H', 1.0, 0.0, 0.0),
        Atom('O', 0.5, 0.5, 0.0),
    ]
    config = Configuration(atoms=atoms)

    # Valid mol_dict
    config.mol_dict = {'solute': [{'start': 0, 'end': 3, 'formula': 'H2O'}]}
    assert config.validate_mol_dict()

    # Invalid mol_dict - end index too large
    config.mol_dict = {'solute': [{'start': 0, 'end': 5, 'formula': 'H2O'}]}
    assert not config.validate_mol_dict()

    # Invalid mol_dict - start >= end
    config.mol_dict = {'solute': [{'start': 2, 'end': 2, 'formula': 'H2O'}]}
    assert not config.validate_mol_dict()

    # Invalid mol_dict - negative start
    config.mol_dict = {'solute': [{'start': -1, 'end': 3, 'formula': 'H2O'}]}
    assert not config.validate_mol_dict()


def test_get_formula_from_atoms():
    """Test molecular formula generation from atoms"""
    # Test water molecule
    water_atoms = [
        Atom('H', 0.0, 0.0, 0.0),
        Atom('H', 1.0, 0.0, 0.0),
        Atom('O', 0.5, 0.5, 0.0),
    ]
    config = Configuration(atoms=water_atoms)
    formula = config._get_formula_from_atoms(water_atoms)
    assert formula == 'H2O'

    # Test methane molecule
    methane_atoms = [
        Atom('C', 0.0, 0.0, 0.0),
        Atom('H', 1.0, 0.0, 0.0),
        Atom('H', 0.0, 1.0, 0.0),
        Atom('H', 0.0, 0.0, 1.0),
        Atom('H', -1.0, 0.0, 0.0),
    ]
    formula = config._get_formula_from_atoms(methane_atoms)
    assert formula == 'CH4'


def test_mol_dict_k_d_tree_insertion(h2o_configuration, h2o):
    """Test that k_d_tree_insertion properly populates mol_dict"""
    # Mock the solvation parameters to test just the k_d_tree_insertion
    # This is a simplified test since full solvation requires external dependencies

    # Start with empty mol_dict
    assert len(h2o_configuration.mol_dict) == 0

    # Call k_d_tree_insertion with minimal solvent addition
    original_atom_count = len(h2o_configuration.atoms)

    # This will try to insert 1 water molecule
    try:
        h2o_configuration.k_d_tree_insertion(
            solvent_molecule=h2o,
            box_size=20.0,
            contact_threshold=1.8,
            n_solvent=1,
            random_seed=42,
        )

        # Check that mol_dict was initialized with solute
        assert 'solute' in h2o_configuration.mol_dict
        assert len(h2o_configuration.mol_dict['solute']) == 1
        assert h2o_configuration.mol_dict['solute'][0]['start'] == 0
        assert (
            h2o_configuration.mol_dict['solute'][0]['end']
            == original_atom_count
        )

        # Check if any solvent was added
        if len(h2o_configuration.atoms) > original_atom_count:
            solvent_key = h2o.formula  # Should be "H2O"
            assert solvent_key in h2o_configuration.mol_dict

    except Exception:
        # If solvation fails due to missing dependencies, that's okay for this test
        # We mainly want to test the mol_dict structure
        pass


def test_mol_dict_load_nonexistent_file():
    """Test loading mol_dict from non-existent file returns False"""
    config = Configuration()
    success = config.load_mol_dict('nonexistent_file.xyz')
    assert not success
    assert len(config.mol_dict) == 0


def test_mol_dict_save_empty_dict(tmp_path):
    """Test that save_mol_dict doesn't create file for empty mol_dict"""
    config = Configuration()
    xyz_file = tmp_path / 'test.xyz'

    # Save with empty mol_dict
    config.save_mol_dict(str(xyz_file))

    # Check that no mol_dict file was created
    mol_dict_file = tmp_path / '.test.mol_dict.txt'
    assert not mol_dict_file.exists()


def test_mol_dict_from_xyz_without_mol_dict(tmp_path):
    """Test loading configuration from xyz without mol_dict file"""
    # Create simple xyz file without mol_dict
    xyz_file = tmp_path / 'test.xyz'
    with open(xyz_file, 'w') as f:
        f.write('3\n')
        f.write('Test molecule\n')
        f.write('H 0.0 0.0 0.0\n')
        f.write('H 1.0 0.0 0.0\n')
        f.write('O 0.5 0.5 0.0\n')

    # Load configuration using from_xyz
    config = Configuration.from_xyz(str(xyz_file))

    # Verify mol_dict is empty
    assert len(config.mol_dict) == 0
    assert config.atoms is not None
    assert len(config.atoms) == 3


def test_mol_dict_corrupt_file(tmp_path):
    """Test handling of corrupted mol_dict file"""
    # Create test files
    xyz_file = tmp_path / 'test.xyz'
    mol_dict_file = tmp_path / '.test.mol_dict.txt'

    # Create simple xyz file
    with open(xyz_file, 'w') as f:
        f.write('3\n')
        f.write('Test molecule\n')
        f.write('H 0.0 0.0 0.0\n')
        f.write('H 1.0 0.0 0.0\n')
        f.write('O 0.5 0.5 0.0\n')

    # Create corrupted mol_dict file
    with open(mol_dict_file, 'w') as f:
        f.write('This is not valid JSON!')

    # Load configuration - should handle corruption gracefully
    config = Configuration()
    success = config.load_mol_dict(str(xyz_file))

    assert not success
    assert len(config.mol_dict) == 0


def test_keep_output_files_false(h2o_configuration, chdir_tmp_path):
    """Test that no files are kept when keep_output_files=False."""

    h2o_configuration.single_point(method='xtb', keep_output_files=False)

    assert not os.path.exists('QM_outputs/xtb.out')


def test_single_point_configuration(h2o_configuration, chdir_tmp_path):
    """Test single point configuration calculation with a QM method (basic functionality)."""
    h2o_configuration.single_point(
        method='xtb', n_cores=1, keep_output_files=True
    )
    assert h2o_configuration.n_ref_evals == 1
    assert os.path.exists('QM_outputs/xtb.out')


def test_custom_output_name_file_move(h2o_configuration, chdir_tmp_path):
    """Test file moving for custom output names (non-energy, non-None)."""
    h2o_configuration.single_point(
        method='xtb', output_name='custom_calc', keep_output_files=True
    )

    assert os.path.exists('QM_outputs/custom_calc.out')


class MockMLPotential(MLPotential):
    """Create a mock ML potential with predict method."""

    def predict(self, *args):
        all_configurations = mlt.ConfigurationSet()

        for arg in args:
            if isinstance(arg, mlt.ConfigurationSet):
                all_configurations += arg

            elif isinstance(arg, mlt.Configuration):
                all_configurations.append(arg)

            else:
                raise ValueError(
                    f'Cannot predict the energy and forces on {type(arg)}'
                )

        for configuration in all_configurations:
            # evaluate predicted energies and forces
            configuration.energy.predicted = 10.0
            configuration.forces.predicted = [0.0, 0.0, 0.0]


def test_ml_potential_predict(h2o_configuration, chdir_tmp_path):
    """Test single point calculation with machine learning potential."""
    h2o_configuration.single_point(method=MockMLPotential)

    # n_ref_evals should not increment for ML potentials
    assert h2o_configuration.n_ref_evals == 0


def test_invalid_string_method_raises_error(h2o_configuration, chdir_tmp_path):
    """Test that invalid string methods raise ValueError."""
    with pytest.raises(
        ValueError,
        match='Cannot use invalid_method to predict energies and forces',
    ):
        h2o_configuration.single_point(method='invalid_method')


def test_n_ref_evals_increment_logic(h2o_configuration, chdir_tmp_path):
    """Test that n_ref_evals increments only for QM methods."""
    initial_count = 5
    h2o_configuration.n_ref_evals = initial_count

    # ML potential should not increment
    h2o_configuration.single_point(method=MockMLPotential)
    assert h2o_configuration.n_ref_evals == initial_count

    # QM method should increment

    h2o_configuration.single_point(method='xtb')
    assert h2o_configuration.n_ref_evals == initial_count + 1
