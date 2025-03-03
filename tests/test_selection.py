import os
import mlptrain as mlt
from autode.atoms import Atom
from mlptrain.training.selection import AtomicEnvSimilarity, AbsDiffE, AtomicEnvDistance
from mlptrain.descriptors import SoapDescriptor

here = os.path.abspath(os.path.dirname(__file__))

def _similar_methane():
    atoms = [
        Atom('C', -0.83511, 2.41296, 0.00000),
        Atom('H', 0.24737, 2.41296, 0.00000),
        Atom('H', -1.19178, 2.07309, 0.94983),
        Atom('H', -1.19178, 1.76033, -0.76926),
        Atom('H', -1.28016, 3.36760, -0.18057),
    ]
    return mlt.Configuration(atoms=atoms)

def _distorted_methane():
    atoms = [
        Atom('C', -0.83511, 2.41296, 0.00000),
        Atom('H', 0.34723, 2.42545, 0.00000),
        Atom('H', -1.19178, 2.07309, 0.94983),
        Atom('H', -1.50592, -0.01979, -0.76926),
        Atom('H', -1.28016, 3.36760, -0.18057),
    ]
    return mlt.Configuration(atoms=atoms)

def test_selection_on_structures():
    configs = mlt.ConfigurationSet()
    file_path = os.path.join(here, 'data', 'methane.xyz')
    configs.load_xyz(filename=file_path, charge=0, mult=1, box=None)
    assert len(configs) == 3

    for avg_mode in ['inner', 'outer', None]:
        soap_descriptor = SoapDescriptor(elements=['C', 'H'], average=avg_mode)
        selector = AtomicEnvSimilarity(threshold=0.9)
        mlp = mlt.potentials.GAP('blank')
        mlp.training_data = configs

        selector(configuration=_similar_methane(), mlp=mlp)
        assert not selector.select

        selector(configuration=_distorted_methane(), mlp=mlp)
        assert selector.select

def test_soap_representation_with_selection():
    """Test if different SoapDescriptor settings work with selection methods."""
    configs = mlt.ConfigurationSet()
    file_path = os.path.join(here, 'data', 'methane.xyz')
    configs.load_xyz(filename=file_path, charge=0, mult=1, box=None)
    mlp = mlt.potentials.GAP('blank')
    mlp.training_data = configs

    for avg_mode in ['inner', 'outer', None]:
        soap_descriptor = SoapDescriptor(elements=['C', 'H'], average=avg_mode)
        abs_diff_selector = AbsDiffE(e_thresh=0.05)
        abs_diff_selector(configuration=_distorted_methane(), mlp=mlp, method_name='DFT')
        assert abs_diff_selector.select or abs_diff_selector.too_large

        env_distance_selector = AtomicEnvDistance()
        env_distance_selector(configuration=_distorted_methane(), mlp=mlp)
        assert env_distance_selector.select == (mlp.n_train > 30)
