import os
import numpy as np

import mlptrain as mlt
from mlptrain.utils import work_in_tmp_dir
from .test_potential import TestPotential
from .molecules import _h2

mlt.Config.n_cores = 1
here = os.path.abspath(os.path.dirname(__file__))


def _get_avg_dists(atoms, atom_pair_list):
    """Return the average distance between atoms in all m pairs"""

    euclidean_dists = [
        atoms.get_distance(i, j, mic=True) for (i, j) in atom_pair_list
    ]

    return np.mean(euclidean_dists)


@work_in_tmp_dir()
def test_bias():

    system = mlt.System(_h2(), box=[50, 50, 50])
    pot = TestPotential("1D")

    config = system.random_configuration()

    bias = mlt.Bias(mlt.AverageDistance([0, 1]), reference=0.7, kappa=100)

    assert bias.ref is not None
    assert bias.kappa is not None
    assert bias.f.atom_pair_list == [(0, 1)]

    new_pos = [[0, 0, 0], [0, 0, 1]]

    ase_atoms = config.ase_atoms
    ase_atoms.set_positions(new_pos, apply_constraint=False)

    assert np.isclose(bias(ase_atoms), 4.5)  # (kappa / 2) * (1-0.7)^2

    bias_force = bias.grad(ase_atoms)

    assert bias_force[0][2] == -bias_force[1][[2]]
    assert np.isclose(bias_force[0][2], -30)  # kappa * (1-0.7)

    trajectory = mlt.md.run_mlp_md(
        configuration=config,
        mlp=pot,
        fs=1000,
        temp=300,
        dt=0.5,
        interval=10,
        bias=bias,
    )

    data = [
        _get_avg_dists(config.ase_atoms, [[0, 1]]) for config in trajectory
    ]

    hist, bin_edges = np.histogram(data, density=False, bins=500)
    mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    mean_value = np.average(mids, weights=hist)

    assert np.isclose(mean_value, 0.7, 0.1)

    trajectory.save_xyz("tmp.xyz")

    assert os.path.exists("tmp.xyz")
