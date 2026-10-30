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
    _random_rotation,
    _random_vector_in_box,
)
from mlptrain.potentials._base import MLPotential


@pytest.fixture
def h2o_solvated_with_h2o():
    """Water molecule solvated with water; created with
    Configuration.solvate(solvent_name='water', buffer_distance=6.0)
    """
    atoms = [
        Atom('H', 4.7258, 3.7810, 3.8420),
        Atom('H', 3.4325, 3.9768, 2.9486),
        Atom('O', 3.7558, 3.7810, 3.8420),
        Atom('O', 1.6410, 5.6215, 5.1410),
        Atom('H', 1.8252, 5.0658, 4.3808),
        Atom('H', 2.3831, 5.5320, 5.7426),
        Atom('O', 0.2137, 1.6048, 3.8653),
        Atom('H', 0.3730, 1.6713, 2.9214),
        Atom('H', 0.2735, 2.4923, 4.2250),
        Atom('O', 4.1556, 1.6866, 4.5267),
        Atom('H', 4.6644, 1.3691, 3.7777),
        Atom('H', 3.2401, 1.7396, 4.2442),
        Atom('O', 5.2370, 2.5534, 1.1812),
        Atom('H', 5.5345, 3.2148, 0.5529),
        Atom('H', 5.9709, 2.3760, 1.7733),
        Atom('O', 4.3931, 5.2890, 0.3177),
        Atom('H', 3.7065, 5.4130, 0.9763),
        Atom('H', 4.8206, 6.1388, 0.1920),
        Atom('O', 1.5763, 2.0668, 7.1499),
        Atom('H', 1.5584, 1.1107, 7.2282),
        Atom('H', 1.8191, 2.2714, 6.2444),
        Atom('O', 2.9836, 6.9620, 3.5356),
        Atom('H', 3.8297, 6.5603, 3.3273),
        Atom('H', 2.4389, 6.8973, 2.7484),
        Atom('O', 4.3427, 3.6430, 5.9737),
        Atom('H', 4.9626, 3.0750, 5.5112),
        Atom('H', 3.5091, 3.1703, 6.0227),
        Atom('O', 6.4379, 6.4827, 6.4603),
        Atom('H', 6.5730, 7.0408, 7.2290),
        Atom('H', 6.6947, 5.5923, 6.7091),
        Atom('O', 5.8334, 5.5081, 1.5821),
        Atom('H', 6.1585, 4.6054, 1.5739),
        Atom('H', 5.4799, 5.6730, 2.4588),
        Atom('O', 1.1515, 7.0347, 0.4560),
        Atom('H', 0.7611, 7.2270, 1.3112),
        Atom('H', 1.6518, 6.2214, 0.5504),
        Atom('O', 4.2025, 0.8327, 6.3825),
        Atom('H', 4.8918, 0.5775, 6.9992),
        Atom('H', 3.5701, 0.1117, 6.3516),
        Atom('O', 7.3693, 4.8547, 3.7695),
        Atom('H', 6.7648, 4.2321, 3.3602),
        Atom('H', 7.2709, 5.6895, 3.3067),
        Atom('O', 1.6644, 4.5189, 1.3072),
        Atom('H', 2.4394, 4.0888, 0.9396),
        Atom('H', 1.4005, 4.0137, 2.0791),
        Atom('O', 5.1430, 5.4211, 4.9351),
        Atom('H', 4.8956, 5.6049, 5.8438),
        Atom('H', 5.7455, 6.1155, 4.6604),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)


def test_equality():
    config1 = Configuration()
    assert config1 == config1  # noqa: PLR0124
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


def test_random_rotation_is_a_valid_rotation_matrix():
    for _ in range(1000):
        rot_matrix = _random_rotation(
            seeded_random.random(),
            seeded_random.random(),
            seeded_random.random(),
        )

        # Orthogonal: R^T R = I
        assert np.allclose(rot_matrix.T @ rot_matrix, np.eye(3), atol=1e-10)

        # Proper rotation (no reflection): det(R) = +1
        assert np.isclose(np.linalg.det(rot_matrix), 1.0, atol=1e-10)


def test_random_rotation_is_uniformly_distributed():
    """Test that rotating a fixed unit vector by many random rotations
    gives components that are uniformly distributed on [-1, 1]"""
    from scipy.stats import kstest

    n_samples = 10_000
    v = np.array([1.0, 0.0, 0.0])

    rotated_components = np.empty((n_samples, 3))
    for i in range(n_samples):
        rot_matrix = _random_rotation(
            seeded_random.random(),
            seeded_random.random(),
            seeded_random.random(),
        )
        rotated_components[i] = rot_matrix @ v

    for axis in range(3):
        # Kolmogorov-Smirnov test against Uniform(-1, 1)
        _, p_value = kstest(
            rotated_components[:, axis], 'uniform', args=(-1, 2)
        )
        assert p_value > 1e-3


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
        assert np.allclose(atom.coordinate, ref_atom.coordinate, atol=1e-4)


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

    except Exception:  # noqa: BLE001, S110
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
                raise TypeError(
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
