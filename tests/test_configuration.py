from autode.atoms import Atom
from mlptrain.configurations.configuration import (
    Configuration,
    random_vector_in_box,
    get_max_mol_distance,
)


def test_equality():
    config1 = Configuration()
    assert config1 == config1
    assert config1 == Configuration()

    config2 = Configuration(atoms=[Atom('H')])

    assert config1 != config2


def test_random_vector_in_box():
    vector = random_vector_in_box(10)
    assert vector <= 10
    assert vector >= 0


def test_get_max_mol_distance(h2o_configuration):
    max_distance_h2o = get_max_mol_distance(h2o_configuration.atoms)
    max_distance_h2o = round(max_distance_h2o, 2)
    assert max_distance_h2o == 1.584
