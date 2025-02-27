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
from repos.mlp_train_fix_table_issue.tests.conftest import h2o


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


def test_wrong_solvent_name_raises_not_found():
    with pytest.raises(SolventNotFound):
        h2o.solvate(solvent_name='solvo_solverson')


def test_no_inputs_for_solvate():
    with pytest.raises(ValueError):
        h2o.solvate()


def test_only_molecule_for_solvate():
    with pytest.raises(ValueError):
        h2o.solvate(molecule=h2o)


def test_only_density_for_solvate():
    with pytest.raises(ValueError):
        h2o.solvate(solvent_density=1)


def test_only_too_many_inputs_for_solvate():
    with pytest.raises(ValueError):
        h2o.solvate(solvent_name='water', solvent_density=1, molecule=h2o)


def test_negative_density_for_solvate():
    with pytest.raises(ValueError):
        h2o.solvate(solvent_name='water', solvent_density=-1)


def test_no_atoms_in_solvent_molecule(empty_molecule):
    with pytest.raises(ValueError):
        h2o.solvate(solvent_density=1, molecule=empty_molecule)
