import os
import mlptrain as mlt
from autode.atoms import Atom
from mlptrain.descriptor import SoapDescriptor
from mlptrain.training.selection import AtomicEnvSimilarity


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

    SoapDescriptor1 = mlt.descriptor.SoapDescriptor(
        average='outer', r_cut=6.0, n_max=8, l_max=8
    )
    SoapDescriptor2 = mlt.descriptor.SoapDescriptor(
        average='inner', r_cut=5.0, n_max=6, l_max=6
    )

    selector1 = AtomicEnvSimilarity(descriptor=SoapDescriptor1, threshold=0.9)
    selector2 = AtomicEnvSimilarity(descriptor=SoapDescriptor2, threshold=0.9)

    mlp = mlt.potentials.GAP('blank')
    mlp.training_data = configs

    selector1(configuration=_similar_methane(), mlp=mlp)
    selector2(configuration=_similar_methane(), mlp=mlp)

    assert not selector1.select
    assert not selector2.select

    selector1(configuration=_distorted_methane(), mlp=mlp)
    selector2(configuration=_distorted_methane(), mlp=mlp)

    assert selector1.select
    assert selector2.select


def test_outlier_identifier():
    configs = mlt.ConfigurationSet()

    file_path = os.path.join(here, 'data', 'methane.xyz')
    configs.load_xyz(filename=file_path, charge=0, mult=1, box=None)

    descriptor = SoapDescriptor(average='outer', r_cut=6.0, n_max=8, l_max=8)

    mlp = mlt.potentials.GAP('blank')
    mlp.training_data = configs

    # Similar configuration should not be an outlier
    result1 = mlt.training.selection._outlier_identifier(
        configuration=_similar_methane(),
        configurations=mlp.training_data,
        descriptor=descriptor,
        dim_reduction=False,
    )
    assert result1 == 1

    # Distorted configuration should be an outlier
    result2 = mlt.training.selection._outlier_identifier(
        configuration=_distorted_methane(),
        configurations=mlp.training_data,
        descriptor=descriptor,
        dim_reduction=False,
    )
    assert result2 == -1
