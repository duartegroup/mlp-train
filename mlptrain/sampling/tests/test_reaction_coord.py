import numpy as np
import pytest

import mlptrain as mlt
from ase.atoms import Atoms as ASEAtoms
from .molecules import _h2o


def test_differencedistance():
    """Test the DifferenceDistance class for reaction coordinate"""

    system = mlt.System(_h2o(), box=[50, 50, 50])

    config = system.random_configuration()
    atoms = config.ase_atoms

    diff_dist = mlt.DifferenceDistance((0, 1), (0, 2))

    # Reaction coordinate should contain two pairs of atoms
    assert len(diff_dist.atom_pair_list) == 2

    rxn_coord = diff_dist(atoms)

    # Check calling the class returns the reaction coordinate
    assert np.isclose(rxn_coord, 0.614, 0.1)

    grad = diff_dist.grad(atoms)

    # Check gradient is close to expected gradient
    assert np.isclose(grad[0][0], -0.1835, 0.1)

    # Gradient matrix should consist of N atoms multiplied by 3 (x, y, z)
    assert grad.shape == (len(atoms), 3)

    # mlp-train should raise a ValueError when != 2 atoms are specified
    with pytest.raises(ValueError):
        mlt.DifferenceDistance((0, 1, 2), (0, 1))

    # mlp-train should raise a ValueError when != 2 pairs are specified
    with pytest.raises(ValueError):
        mlt.DifferenceDistance((0, 1))


@pytest.mark.parametrize(
    'rs', [[(0, 1), (0, 2)], [(1, 0), (0, 2)], [(1, 0), (2, 0)]]
)
def test_differencedistance_numerical_gradient(rs, h=1e-8):
    """Test that the analytic gradient is correct for differencedistance"""

    atoms = ASEAtoms(
        symbols=['H', 'H', 'H'],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.1, 0.3], [-2.0, 0.2, 0.4]],
    )

    z = mlt.DifferenceDistance(*rs)
    grad = z.grad(atoms)
    e = z(atoms)

    for i in range(3):
        for j in range(3):
            # Shift to a new position, evaluate the energy and shift back
            atoms.positions[i, j] += h
            e_plus_h = z(atoms)
            atoms.positions[i, j] -= h

            num_grad_ij = (e_plus_h - e) / h

            assert np.isclose(grad[i, j], num_grad_ij, atol=1e-8)
