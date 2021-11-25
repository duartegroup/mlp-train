import os
import numpy as np
from autode.atoms import Atom
from mltrain.configurations import ConfigurationSet, Configuration
from mltrain.utils import work_in_tmp_dir
from mltrain.box import Box


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

    configs = ConfigurationSet(Configuration(atoms=[Atom('H')],
                                             charge=-1,
                                             mult=3,
                                             box=Box([1., 1., 1.])))
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

    for attr in ('energy', 'forces'):
        for kind in ('predicted', 'true'):

            assert np.allclose(getattr(getattr(loaded_config, attr), kind),
                               getattr(getattr(config, attr), kind))


@work_in_tmp_dir()
def test_configurations_load_xyz():

    configs = ConfigurationSet()

    with open('tmp.xyz', 'w') as xyz_file:
        print('1',
              'title line',
              'H   0.0   0.0   0.0',
              '1',
              'title line',
              'H   1.0   0.0   0.0',
              sep='\n', file=xyz_file)

    configs.load_xyz('tmp.xyz', charge=0, mult=2)

    assert len(configs) == 2
    for config in configs:
        assert config.charge == 0
        assert config.mult == 2


def test_remove_energy_above():

    c1 = Configuration(atoms=[Atom('H'), Atom('H', x=0.7)])
    c1.energy.true = -1.0

    c2 = Configuration(atoms=[Atom('H'), Atom('H', x=0.5)])
    c2.energy.true = -0.5

    configs = ConfigurationSet(c1, c2)

    # With a threshold of 1 eV both configurations should remain after pruning
    configs.remove_above_e(1.0)
    print(configs[0].energy.true)
    assert len(configs) == 2

    # but with a tighter 0.2 eV threshold it should remove the higher one
    configs.remove_above_e(0.2)
    assert len(configs) == 1

    assert np.isclose(configs[0].energy.true, -1.0, atol=1E-10)
