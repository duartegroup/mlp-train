import os
import numpy as np
import mltrain as mlt
from autode.atoms import Atom
from mltrain.potentials._base import MLPotential
from ase.calculators.calculator import Calculator
from mltrain.utils import work_in_tmp_dir
mlt.Config.n_cores = 1
here = os.path.abspath(os.path.dirname(__file__))


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

        force = - 2 * derivative * (r-0.7)

        return force


class TestPotential(MLPotential):

    __test__ = False

    def __init__(self,
                 name: str,
                 system=None):

        super().__init__(name=name, system=system)

    @property
    def ase_calculator(self):

        potential = HarmonicPotential()

        return potential

    def _train(self) -> None:
        """Train this potential on self._training_data"""

    def requires_atomic_energies(self) -> bool:
        """Does this potential need E_0s for each atom to be specified"""

    def requires_non_zero_box_size(self) -> bool:
        """Can this potential be run in a box with side lengths = 0"""


@work_in_tmp_dir()
def test_bias():

    system = mlt.System(_h2(), box=[50, 50, 50])
    pot = TestPotential('1D')

    config = system.random_configuration()

    bias = mlt.Bias(to_average=[[0, 1]], reference=2, kappa=10)
    
    assert bias.ref is not None
    assert bias.kappa is not None
    assert bias.rxn_coord == [[0, 1]]

    new_pos = [[0, 0, 0],
               [0, 0, 1]]

    ase_atoms = config.ase_atoms
    ase_atoms.set_positions(new_pos, apply_constraint=False)

    bias_energy = bias.__call__([[0, 1]], ase_atoms)

    assert bias_energy == 5  # (kappa / 2) * (1-2)^2

    bias_force = bias.grad([[0, 1]], ase_atoms)

    assert bias_force[0][2] == - bias_force[1][[2]]
    assert bias_force[0][2] == 10

    trajectory = mlt.md.run_mlp_md(configuration=config,
                                   mlp=pot,
                                   fs=100,
                                   temp=300,
                                   dt=0.5,
                                   interval=10,
                                   bias=bias)

    trajectory.save_xyz('tmp.xyz')

    assert os.path.exists('tmp.xyz')


@work_in_tmp_dir()
def test_window_umbrella():

    charge, mult = -1, 1

    system = mlt.System(_h2(), box=[50, 50, 50])
    pot = TestPotential('1D')

    config = system.random_configuration()
    new_pos = [[0, 0, 0],
               [0, 0, 1]]

    ase_atoms = config.ase_atoms
    ase_atoms.set_positions(new_pos, apply_constraint=False)

    mean_distances = mlt.umbrella._get_rxn_coords(ase_atoms, [[0, 1]])

    assert mean_distances == 1.0

    umbrella = mlt.UmbrellaSampling(to_average=[[0, 1]], kappa=10)

    assert umbrella.kappa is not None
    assert umbrella.num_pairs == 1
    assert umbrella.refs is None

    traj = mlt.ConfigurationSet()
    traj.load_xyz(os.path.join(here, 'data', 'h2_traj.xyz'),
                  charge=charge,
                  mult=mult)

    _ = umbrella._get_window_frames(traj, num_windows=10,
                                    init_ref=0.7, final_ref=2)

    assert np.alltrue(umbrella.refs == np.linspace(0.7, 2, 10))

    umbrella.run_umbrella_sampling(traj,
                                   pot,
                                   temp=300,
                                   interval=5,
                                   dt=0.5,
                                   num_windows=3,
                                   fs=500)

    assert umbrella.bias.kappa == 10
    assert umbrella.refs is not None

    assert os.path.exists('combined_windows.xyz')
    assert os.path.exists('fitted_data.pdf')
