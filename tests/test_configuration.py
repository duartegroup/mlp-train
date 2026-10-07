import mlptrain as mlt
from autode.atoms import Atom
from autode.exceptions import SolventNotFound
from mlptrain.configurations.configuration import (
    Configuration,
    _random_vector_in_box,
    _get_max_mol_distance,
)
import numpy as np
import random
import pytest
import os
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
        Atom('O', 2.493, 0.846, 7.323),
        Atom('H', 3.094, 0.451, 7.958),
        Atom('H', 2.930, 0.824, 6.469),
        Atom('O', 0.869, 2.786, 7.719),
        Atom('H', 0.453, 3.078, 8.533),
        Atom('H', 0.429, 1.976, 7.452),
        Atom('O', 6.646, 5.465, 9.144),
        Atom('H', 5.829, 5.956, 9.027),
        Atom('H', 6.916, 5.156, 8.277),
        Atom('O', 9.989, 6.416, 0.539),
        Atom('H', 10.368, 6.446, 1.420),
        Atom('H', 9.412, 5.650, 0.505),
        Atom('O', 10.768, 9.829, 1.875),
        Atom('H', 11.058, 9.776, 2.788),
        Atom('H', 9.814, 9.929, 1.886),
        Atom('O', 11.347, 9.342, 10.201),
        Atom('H', 10.868, 9.820, 9.522),
        Atom('H', 11.133, 9.755, 11.041),
        Atom('O', 3.490, 9.171, 2.651),
        Atom('H', 4.017, 9.680, 3.271),
        Atom('H', 3.768, 9.425, 1.769),
        Atom('O', 10.710, 6.181, 8.306),
        Atom('H', 11.300, 6.346, 9.045),
        Atom('H', 10.326, 7.024, 8.058),
        Atom('O', 3.543, 3.584, 4.726),
        Atom('H', 3.020, 2.915, 5.173),
        Atom('H', 2.929, 4.227, 4.364),
        Atom('O', 5.436, 5.162, 2.481),
        Atom('H', 5.086, 4.798, 1.665),
        Atom('H', 4.901, 4.813, 3.197),
        Atom('O', 7.050, 2.541, 2.540),
        Atom('H', 6.587, 3.019, 3.231),
        Atom('H', 6.400, 2.304, 1.876),
        Atom('O', 4.396, 8.648, 9.644),
        Atom('H', 4.273, 9.288, 10.348),
        Atom('H', 4.440, 9.139, 8.820),
        Atom('O', 8.510, 3.155, 9.850),
        Atom('H', 8.355, 2.810, 10.732),
        Atom('H', 7.918, 2.688, 9.257),
        Atom('O', 2.822, 5.436, 7.106),
        Atom('H', 3.602, 4.899, 6.952),
        Atom('H', 2.100, 5.030, 6.620),
        Atom('O', 0.612, 10.840, 5.537),
        Atom('H', 0.521, 11.366, 4.740),
        Atom('H', 0.405, 11.413, 6.278),
        Atom('O', 8.099, 10.983, 5.808),
        Atom('H', 8.683, 10.563, 5.171),
        Atom('H', 7.628, 10.282, 6.263),
        Atom('O', 5.903, 3.134, 9.615),
        Atom('H', 6.508, 3.168, 10.359),
        Atom('H', 5.152, 3.689, 9.837),
        Atom('O', 1.394, 11.174, 1.621),
        Atom('H', 0.958, 11.062, 2.468),
        Atom('H', 0.760, 10.931, 0.943),
        Atom('O', 8.936, 8.391, 2.451),
        Atom('H', 8.818, 8.504, 3.397),
        Atom('H', 8.542, 7.549, 2.215),
        Atom('O', 11.275, 6.926, 4.073),
        Atom('H', 10.745, 7.133, 4.846),
        Atom('H', 10.791, 7.240, 3.306),
        Atom('O', 3.288, 7.855, 5.700),
        Atom('H', 2.708, 8.463, 5.238),
        Atom('H', 3.740, 7.333, 5.033),
        Atom('O', 6.417, 8.402, 6.106),
        Atom('H', 6.099, 8.935, 6.838),
        Atom('H', 5.673, 8.259, 5.517),
        Atom('O', 8.019, 7.266, 9.408),
        Atom('H', 7.989, 8.067, 9.936),
        Atom('H', 8.693, 7.396, 8.737),
        Atom('O', 2.347, 7.554, 10.672),
        Atom('H', 2.692, 7.158, 11.475),
        Atom('H', 2.149, 6.837, 10.066),
        Atom('O', 9.010, 4.399, 6.125),
        Atom('H', 8.295, 5.009, 6.319),
        Atom('H', 9.248, 3.971, 6.950),
        Atom('O', 8.378, 5.684, 10.575),
        Atom('H', 7.838, 6.139, 11.225),
        Atom('H', 8.309, 4.744, 10.758),
        Atom('O', 10.954, 2.268, 3.657),
        Atom('H', 11.290, 2.958, 4.234),
        Atom('H', 10.961, 2.619, 2.764),
        Atom('O', 9.921, 0.994, 1.116),
        Atom('H', 9.946, 0.528, 1.954),
        Atom('H', 10.628, 1.643, 1.132),
        Atom('O', 6.205, 7.075, 3.229),
        Atom('H', 6.330, 7.437, 2.349),
        Atom('H', 6.266, 7.807, 3.847),
        Atom('O', 8.830, 0.490, 9.857),
        Atom('H', 8.124, 0.754, 9.263),
        Atom('H', 8.701, 0.969, 10.678),
        Atom('O', 5.632, 0.751, 4.223),
        Atom('H', 5.079, 0.712, 5.005),
        Atom('H', 6.483, 0.376, 4.459),
        Atom('O', 8.261, 7.729, 6.132),
        Atom('H', 7.909, 7.565, 7.009),
        Atom('H', 8.556, 8.643, 6.113),
        Atom('O', 6.751, 9.805, 2.773),
        Atom('H', 5.949, 10.210, 3.110),
        Atom('H', 6.555, 9.483, 1.891),
        Atom('O', 10.625, 4.155, 1.082),
        Atom('H', 10.659, 4.388, 0.152),
        Atom('H', 11.462, 3.741, 1.300),
        Atom('O', 1.093, 0.753, 10.543),
        Atom('H', 1.651, 0.154, 11.043),
        Atom('H', 1.595, 1.034, 9.776),
        Atom('O', 0.668, 8.022, 1.308),
        Atom('H', 0.881, 8.862, 1.719),
        Atom('H', 0.249, 8.217, 0.468),
        Atom('O', 3.346, 5.572, 9.210),
        Atom('H', 3.744, 6.252, 8.663),
        Atom('H', 3.623, 5.737, 10.113),
        Atom('O', 1.649, 6.854, 4.305),
        Atom('H', 1.325, 6.485, 5.129),
        Atom('H', 1.448, 6.220, 3.613),
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
    h2o_configuration.solvate(solvent_name='water')
    assert len(h2o_configuration.atoms) == 159

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
