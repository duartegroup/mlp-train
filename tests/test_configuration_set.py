import os
import numpy as np
from autode.atoms import Atom
from mlptrain.configurations import ConfigurationSet, Configuration
from mlptrain.utils import work_in_tmp_dir
from mlptrain.box import Box


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


@work_in_tmp_dir()
def test_configurations_load_xyz_with_energies_forces():
    """
    Test loading an xyz file for a config set wth variable energies,
    forces and box sizes.
    """
    configs = ConfigurationSet()

    with open('tmp.xyz', 'w') as xyz_file:
        print(
            '22',
            'Lattice="20.000000 0.000000 0.000000 0.000000 20.000000 0.000000 0.000000 0.000000 20.000000" '
            'energy=-11580.70167936 Properties=species:S:1:pos:R:3:forces:R:3',
            'C   0.15752   1.16848   1.24910   -0.98713   -0.42563   -0.93123',
            'C  -0.79522   0.20206   1.91812   -2.38781    2.65485   -0.38040',
            'Be -0.11423  -0.90974   2.23136    1.95299   -1.96673   -0.00367',
            'H  -1.72610   0.53596   2.41821    0.87022   -0.20286   -0.26829',
            'C   1.07995  -0.98564   1.49000   -1.40103    2.61406   -0.29833',
            'H  -0.47282  -1.70552   2.82842   -0.40060   -0.59030    0.54244',
            'C   1.11098   0.20418   0.69541    1.32925   -2.54027    0.51155',
            'H   1.74594  -1.79632   1.48718    0.74392   -0.80923   -0.20624',
            'H   1.96423   0.41181   0.07610    0.06627    0.43665   -0.17306',
            'H   0.65443   1.73573   2.00077    0.45054    0.66643    0.72471',
            'H  -0.25306   1.81109   0.46810   -0.11523    0.08179    0.46247',
            'C  -0.50379  -0.50370  -0.80571   -0.73200   -0.43413    1.44966',
            'C  -1.60911  -0.39092   0.07084    1.57166    0.63002   -1.30423',
            'H  -2.21775   0.47138  -0.06099   -0.36586    0.62542   -0.02913',
            'H  -2.12759  -1.23566   0.40164   -0.46781   -1.15900    0.47771',
            'C  -0.17754   0.57296  -1.75321    1.69309   -3.49468   -0.72671',
            'H  -0.03367  -1.48400  -0.92261   -0.15960    0.52124   -0.26696',
            'O  -0.64023   1.65497  -1.71141   -1.17861    2.90807    0.06166',
            'C   0.83922   0.14416  -2.83825   -1.44421   -0.12439    0.41955',
            'H   1.17303   1.03489  -3.32904    0.44566    0.30180   -0.40030',
            'H   0.31973  -0.52220  -3.55093    0.22722    0.31219    0.27423',
            'H   1.62607  -0.41397  -2.36310    0.46565   -0.16544    0.01173',
            '18',
            'Lattice="18.000000 0.000000 0.000000 0.000000 18.000000 0.000000 0.000000 0.000000 18.000000" '
            'energy=-11581.02323085 Properties=species:S:1:pos:R:3:forces:R:3',
            'C  -0.14388   0.58514   1.72503   -0.26651    0.13421    1.07021',
            'C   0.04181  -0.85624   1.27377   -1.15828    0.33661    0.32617',
            'C   1.30547  -1.19982   1.43121    2.41677    0.72934    0.38652',
            'H  -0.77023  -1.47035   0.85124    0.66526    0.54979    0.49939',
            'C   2.09489  -0.00345   1.95796   -1.16435   -0.35668    0.47563',
            'H   1.80942  -2.13489   1.19406   -0.38060    0.13724    0.00064',
            'C   1.25045   1.00558   2.20650   -1.11760   -2.07214   -1.44189',
            'H   3.11668   0.10081   2.28762    0.28187   -0.61557   -0.35632',
            'H   1.56030   1.92262   2.59835    0.05989    1.31003    0.53325',
            'H  -0.95605   0.73278   2.48750    0.81396   -0.33790   -0.40133',
            'H  -0.50535   1.20087   0.91714    0.01837    0.30916   -0.51288',
            'C  -0.86571   1.19876  -2.16169   -2.75334    2.99743    0.65360',
            'C  -1.83154   2.15751  -1.85043    1.55884   -2.59191   -1.02838',
            'H  -2.66091   1.87770  -1.19443    0.59674    0.16747    0.12963',
            'H  -1.91392   3.10727  -2.37756    0.24926   -0.27902    0.12722',
            'C  -0.83439  -0.11351  -1.50115   -0.63239    0.41365   -0.46196',
            'H  -0.18981   1.36454  -2.96508    0.72714    0.38914   -0.29681',
            'O  -1.79851  -0.44558  -0.83933    0.63063   -0.60796   -0.20044',
            sep='\n',
            file=xyz_file,
        )

    configs.load_xyz('tmp.xyz', charge=0, mult=1)

    assert len(configs) == 2

    # check loading properties line
    energies = [-11580.70167936, -11581.02323085]
    num_atoms = [22, 18]
    box_sizes = [(20, 20, 20), (18, 18, 18)]
    for i in range(2):
        config = configs[i]
        assert config.box == Box(box_sizes[i])
        assert config.charge == 0
        assert config.mult == 1
        assert config.energy.true == energies[i]
        assert len(config.atoms) == num_atoms[i]
        assert len(config.coordinates) == num_atoms[i]
        assert len(config.forces.true) > 0

    # check config set force and energy loading
    assert len(configs.true_forces) > 0
    assert len(configs.true_energies) > 0
    assert len(configs._coordinates) > 0
    assert len(configs.plumed_coordinates) > 0
