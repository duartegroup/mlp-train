import os
import numpy as np
import pytest
from autode.atoms import Atom
from mlptrain.configurations import ConfigurationSet, Configuration
from mlptrain.utils import work_in_tmp_dir
from mlptrain.box import Box


@pytest.fixture
def config_set_xyz_with_energies_forces():
    configs = ConfigurationSet()

    with open('tmp.xyz', 'w') as xyz_file:
        print(
            '3',
            'Lattice="20.000000 0.000000 0.000000 0.000000 20.000000 0.000000 0.000000 0.000000 20.000000" '
            'energy=-11580.70167936 Properties=species:S:1:pos:R:3:forces:R:3',
            'C   0.00000   0.00000   0.00000   -1.00000   -1.00000   -1.00000',
            'O   1.00000   1.00000   1.00000   -2.00000    2.00000   -2.00000',
            'H   2.00000   2.00000   2.00000    3.00000   -3.00000   -3.00000',
            '2',
            'Lattice="18.000000 0.000000 0.000000 0.000000 18.000000 0.000000 0.000000 0.000000 18.000000" '
            'energy=-11581.02323085 Properties=species:S:1:pos:R:3:forces:R:3',
            'C   0.00000   0.00000   0.00000    0.00000    0.00000    0.00000',
            'O   1.00000   1.00000  1.00000   -1.00000    1.00000    1.00000',
            sep='\n',
            file=xyz_file,
        )

    configs.load_xyz(
        'tmp.xyz', charge=0, mult=1, load_energies=True, load_forces=True
    )

    expected_values = {
        'energies': [-11580.70167936, -11581.02323085],
        'num_atoms': [3, 2],
        'box_sizes': [(20, 20, 20), (18, 18, 18)],
        'coords': np.array(
            [
                np.array(
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    dtype=float,
                ),
                np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
            ],
            dtype=object,
        ),
        'forces': np.array(
            [
                np.array(
                    [[-1.0, -1.0, -1.0], [-2.0, 2.0, -2.0], [3.0, -3.0, -3.0]],
                    dtype=float,
                ),
                np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]], dtype=float),
            ],
            dtype=object,
        ),
    }

    return configs, expected_values


@work_in_tmp_dir()
def test_configurations_save():
    configs = ConfigurationSet()

    # Configuration sets should be constructable from nothing
    assert len(configs) == 0

    # Will not save blank configurations
    configs.save('tmp')
    assert not os.path.exists('tmp')

    configs.append(Configuration(atoms=[Atom('H')]))
    configs.save('tmp.npz')

    assert os.path.exists('tmp.npz')


@work_in_tmp_dir()
def test_configurations_load_default():
    configs = ConfigurationSet(Configuration(atoms=[Atom('H')]))
    assert len(configs) == 1

    configs.save('tmp.npz')

    new_configs = ConfigurationSet('tmp.npz')

    assert len(new_configs) == 1
    config = new_configs[0]
    assert config.box is None
    assert config.mult == 1
    assert config.charge == 0
    assert config.energy.true == config.energy.predicted is None


@work_in_tmp_dir()
def test_configurations_load_alt_attrs():
    configs = ConfigurationSet(
        Configuration(
            atoms=[Atom('H')], charge=-1, mult=3, box=Box([1.0, 1.0, 1.0])
        )
    )
    configs.save('tmp.npz')
    new_configs = ConfigurationSet('tmp.npz')
    config = new_configs[0]

    assert not config.box.has_zero_volume
    assert config.mult == 3
    assert config.charge == -1
    assert config.energy.true == config.energy.predicted is None


@work_in_tmp_dir()
def test_configurations_load_with_energies_forces():
    config = Configuration(atoms=[Atom('H')])
    config.energy.true = -1.0
    config.energy.predicted = -0.9

    config.forces.true = 1.1 * np.ones(shape=(1, 3))
    config.forces.predicted = 1.105 * np.ones(shape=(1, 3))

    ConfigurationSet(config).save('tmp.npz')
    loaded_config = ConfigurationSet('tmp.npz')[0]

    assert loaded_config.energy.true is not None
    assert loaded_config.energy.predicted is not None
    assert loaded_config.forces.true is not None
    assert loaded_config.forces.predicted is not None

    for attr in ('energy', 'forces'):
        for kind in ('predicted', 'true'):
            assert np.allclose(
                getattr(getattr(loaded_config, attr), kind),
                getattr(getattr(config, attr), kind),
            )


@work_in_tmp_dir()
def test_configurations_load_with_energies_forces_diff_sizes(
    h2o_configuration,
):
    config1 = Configuration(atoms=[Atom('H')])
    config1.energy.true = -1.0
    config1.energy.predicted = -0.9

    config1.forces.true = 1.1 * np.ones(shape=(1, 3))
    config1.forces.predicted = 1.105 * np.ones(shape=(1, 3))

    config2 = h2o_configuration
    config2.energy.true = -3.0
    config2.energy.predicted = -2.8

    config2.forces.true = 1.5 * np.ones(shape=(3, 3))
    config2.forces.predicted = 1.502 * np.ones(shape=(3, 3))

    ConfigurationSet(config1, config2).save('tmp.npz')
    loaded_configs = ConfigurationSet('tmp.npz')

    for config in loaded_configs:
        assert config.energy.true is not None
        assert config.energy.predicted is not None
        assert config.forces.true is not None
        assert config.forces.predicted is not None

    for attr in ('energy', 'forces'):
        for kind in ('predicted', 'true'):
            assert np.allclose(
                getattr(getattr(loaded_configs[0], attr), kind),
                getattr(getattr(config1, attr), kind),
            )


@work_in_tmp_dir()
def test_configurations_load_xyz():
    configs = ConfigurationSet()

    with open('tmp.xyz', 'w') as xyz_file:
        print(
            '1',
            'title line',
            'H   0.0   0.0   0.0',
            '1',
            'title line',
            'H   1.0   0.0   0.0',
            sep='\n',
            file=xyz_file,
        )

    configs.load_xyz('tmp.xyz', charge=0, mult=2)

    assert len(configs) == 2
    for config in configs:
        assert config.charge == 0
        assert config.mult == 2


def test_configurations_load_numpy_compatibility():
    data = ConfigurationSet('data/water_al.npz')

    for config in data:
        assert config.box is not None
        assert config.charge is not None
        assert config.mult is not None
        assert config.energy.true is not None
        assert config.forces.true is not None
        assert config.coordinates is not None


@work_in_tmp_dir()
def test_configurations_load_xyz_with_energies_forces(
    config_set_xyz_with_energies_forces,
):
    # get config set from xyz
    configs, exp_vals = config_set_xyz_with_energies_forces

    assert len(configs) == 2

    # check loading properties line for each config
    for i in range(2):
        config = configs[i]
        assert config.box == Box(exp_vals['box_sizes'][i])
        assert config.charge == 0
        assert config.mult == 1
        assert len(config.atoms) == exp_vals['num_atoms'][i]
        assert config.energy.true == exp_vals['energies'][i]
        assert np.allclose(config.coordinates, exp_vals['coords'][i])
        assert np.allclose(config.forces.true, exp_vals['forces'][i])

    # same tests but for config set properties
    assert all(
        config_energy == exp_vals['energies'][i]
        for i, config_energy in enumerate(configs.true_energies)
    )
    assert all(
        np.allclose(config_coords, exp_vals['coords'][i])
        for i, config_coords in enumerate(configs._coordinates)
    )
    assert all(
        np.allclose(config_forces, exp_vals['forces'][i])
        for i, config_forces in enumerate(configs.true_forces)
    )


@work_in_tmp_dir()
def test_configurations_load_xyz_and_save_npz(
    config_set_xyz_with_energies_forces,
):
    # get config set from xyz
    configs, exp_vals = config_set_xyz_with_energies_forces

    # check saving and loading npz
    configs.save('tmp.npz')
    loaded_configs = ConfigurationSet('tmp.npz')

    # same tests but for config set properties
    assert all(
        config_energy == exp_vals['energies'][i]
        for i, config_energy in enumerate(loaded_configs.true_energies)
    )
    assert all(
        np.allclose(config_coords, exp_vals['coords'][i])
        for i, config_coords in enumerate(loaded_configs._coordinates)
    )
    assert all(
        np.allclose(config_forces, exp_vals['forces'][i])
        for i, config_forces in enumerate(loaded_configs.true_forces)
    )
