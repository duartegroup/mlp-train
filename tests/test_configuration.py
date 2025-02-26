from autode.atoms import Atom
from mlptrain.configurations.configuration import (
    Configuration,
    _random_vector_in_box,
    _get_max_mol_distance,
)
import numpy as np
import random


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
    assert all(
        [
            np.round(atom.coordinate, 3)
            == h2o_solvated_with_h2o.atoms[i].coordinate
            for i, atom in enumerate(h2o_configuration.atoms)
        ]
    )
