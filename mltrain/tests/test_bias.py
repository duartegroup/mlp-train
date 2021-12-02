import os
import mltrain as mlt
import numpy as np

mlt.Config.n_cores = 1


def test_bias():

    system = mlt.System(mlt.Molecule('init.xyz', charge=-1), box=[50, 50, 50])
    gap = mlt.potentials.GAP('sn2', system=system)

    config = system.random_configuration()

    bias = mlt.Bias(to_average=[[0, 1]], reference=2, kappa=100)
    
    assert bias.ref is not None
    assert bias.kappa is not None
    assert bias.rxn_coord == [[0, 1]]

    new_pos = [[0, 0, 0],
               [0, 0, 1],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]
               ]

    ase_atoms = config.ase_atoms
    ase_atoms.set_positions(new_pos, apply_constraint=False)

    bias_energy = bias.__call__([[0, 1]], ase_atoms)

    assert bias_energy == 50  # (kappa / 2) * (1-2)^2

    bias_force = bias.grad([[0, 1]], ase_atoms)

    assert bias_force[0][2] == - bias_force[1][[2]]
    assert bias_force[0][2] == 100

    trajectory = mlt.md.run_mlp_md(configuration=config,
                                   mlp=gap,
                                   fs=100,
                                   temp=300,
                                   dt=0.5,
                                   interval=10,
                                   bias=bias)

    trajectory.save_xyz('sn2.xyz')

    assert os.path.exists('sn2.xyz')


def test_window_umbrella():

    charge, mult = -1, 1

    system = mlt.System(mlt.Molecule('init.xyz', charge=charge),
                        box=[50, 50, 50])
    gap = mlt.potentials.GAP('sn2', system=system)

    config = system.random_configuration()
    new_pos = [[0, 0, 0],
               [0, 0, 1],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]
               ]

    ase_atoms = config.ase_atoms
    ase_atoms.set_positions(new_pos, apply_constraint=False)

    mean_distances = mlt.umbrella._get_rxn_coords(ase_atoms, [[0, 1], [2, 3]])

    assert mean_distances == 0.5

    umbrella = mlt.UmbrellaSampling(to_average=[[0, 1]], kappa=10)

    assert umbrella.kappa is not None
    assert umbrella.num_pairs == 1
    assert umbrella.refs is None

    traj = mlt.ConfigurationSet()
    traj.load_xyz('irc_IRC_Full_trj.xyz', charge=charge, mult=mult)

    _ = umbrella._get_window_frames(traj, num_windows=10,
                                    init_ref=0, final_ref=5)

    assert np.alltrue(umbrella.refs == np.linspace(0, 5, 10))

    umbrella.run_umbrella_sampling(traj,
                                   gap,
                                   temp=300,
                                   interval=5,
                                   dt=0.5,
                                   num_windows=2,
                                   fs=400)

    assert umbrella.bias.kappa == 10
    assert umbrella.refs is not None

    assert os.path.exists('fitted_data.pdf')
