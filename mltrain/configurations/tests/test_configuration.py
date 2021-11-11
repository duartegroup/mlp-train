from autode.atoms import Atom
from mltrain.configurations import Configuration


def test_equality():

    config1 = Configuration()
    assert config1 == config1
    assert config1 == Configuration()

    config2 = Configuration(atoms=[Atom('H')])

    assert config1 != config2

