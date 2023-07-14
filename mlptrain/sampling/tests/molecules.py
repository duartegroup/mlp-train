import mlptrain as mlt
from autode.atoms import Atom


def _h2():
    """Dihydrogen molecule"""
    atoms = [
        Atom("H", -0.80952, 2.49855, 0.0),
        Atom("H", -0.34877, 1.961, 0.0),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)


def _h2o():
    """Water molecule"""
    atoms = [
        Atom("H", 2.32670, 0.51322, 0.0),
        Atom("H", 1.03337, 0.70894, -0.89333),
        Atom("O", 1.35670, 0.51322, 0.0),
    ]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)
