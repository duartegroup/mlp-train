import os
import numpy as np
import mltrain as mlt
from autode.atoms import Atom
from mltrain.potentials._base import MLPotential
from ase.calculators.calculator import Calculator
from mltrain.utils import work_in_tmp_dir
mlt.Config.n_cores = 1
here = os.path.abspath(os.path.dirname(__file__))


def _get_avg_dists(atoms, atom_pair_list):
    """Return the average distance between atoms in all m pairs"""

    euclidean_dists = [atoms.get_distance(i, j, mic=True)
                       for (i, j) in atom_pair_list]

    return np.mean(euclidean_dists)


def _h2():
    """Dihydrogen molecule"""
    atoms = [Atom('H', -0.80952, 2.49855, 0.), Atom('H', -0.34877, 1.961, 0.)]
    return mlt.Molecule(atoms=atoms, charge=0, mult=1)


class HarmonicPotential(Calculator):

    def get_potential_energy(self, atoms):

        r = atoms.get_distance(0, 1)

        return (r - 0.7)**2

    def get_forces(self, atoms):

        derivative = np.zeros((len(atoms), 3))

        r = atoms.get_distance(0, 1)

        x_dist, y_dist, z_dist = [atoms[0].position[j] - atoms[1].position[j]
                                  for j in range(3)]

        x_i, y_i, z_i = (x_dist / r), (y_dist / r), (z_dist / r)

        derivative[0][:] = [x_i, y_i, z_i]
        derivative[1][:] = [-x_i, -y_i, -z_i]

        force = -2 * derivative * (r-0.7)

        return force


class TestPotential(MLPotential):

    __test__ = False

    def __init__(self,
                 name: str,
                 system=None):

        super().__init__(name=name, system=system)

    @property
    def ase_calculator(self):

        return HarmonicPotential()

    def _train(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""

    def requires_atomic_energies(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""

    def requires_non_zero_box_size(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""


@work_in_tmp_dir()
def test_bias():

    system = mlt.System(_h2(), box=[50, 50, 50])
    pot = TestPotential('1D')

    config = system.random_configuration()

    bias = mlt.Bias(mlt.AverageDistance([0, 1]),
                    reference=0.7,
                    kappa=100)
    
    assert bias.ref is not None
    assert bias.kappa is not None
    assert bias.f.atom_pair_list == [(0, 1)]

    new_pos = [[0, 0, 0],
               [0, 0, 1]]

    ase_atoms = config.ase_atoms
    ase_atoms.set_positions(new_pos, apply_constraint=False)

    assert np.isclose(bias(ase_atoms), 4.5)  # (kappa / 2) * (1-0.7)^2

    bias_force = bias.grad(ase_atoms)

    assert bias_force[0][2] == - bias_force[1][[2]]
    assert np.isclose(bias_force[0][2], -30)  # kappa * (1-0.7)

    trajectory = mlt.md.run_mlp_md(configuration=config,
                                   mlp=pot,
                                   fs=1000,
                                   temp=300,
                                   dt=0.5,
                                   interval=10,
                                   bias=bias)

    data = [_get_avg_dists(config.ase_atoms, [[0, 1]])
            for config in trajectory]

    hist, bin_edges = np.histogram(data, density=False, bins=500)
    mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    mean_value = np.average(mids, weights=hist)

    assert np.isclose(mean_value, 0.7, 0.1)

    trajectory.save_xyz('tmp.xyz')

    assert os.path.exists('tmp.xyz')


@work_in_tmp_dir()
def test_window_umbrella():

    umbrella = mlt.UmbrellaSampling(zeta_func=mlt.AverageDistance([0, 1]),
                                    kappa=100)

    assert umbrella.kappa is not None and np.isclose(umbrella.kappa, 100.)
    assert umbrella.refs is None

    traj = mlt.ConfigurationSet()
    traj.load_xyz(os.path.join(here, 'data', 'h2_traj.xyz'), charge=0, mult=1)

    umbrella._set_reference_values(traj, num=10, init_ref=0.7, final_ref=2)

    # Setting the reference values of the reaction coordinate should un-None
    assert umbrella.refs is not None
    assert np.allclose(umbrella.refs, np.linspace(0.7, 2, 10))

    umbrella.run_umbrella_sampling(traj,
                                   mlp=TestPotential('1D'),
                                   temp=300,
                                   interval=5,
                                   dt=0.5,
                                   n_windows=2,
                                   fs=1000)

    # Sampling with a high force constant should lead to fitted Gaussians
    # that closely match the reference (target) values
    for gaussian, ref in zip(umbrella._fitted_gaussians, umbrella.refs):
        assert np.isclose(gaussian.params[1], ref, atol=0.1)

    assert os.path.exists('combined_windows.xyz')
    assert os.path.exists('fitted_data.pdf')

    umbrella.wham()
