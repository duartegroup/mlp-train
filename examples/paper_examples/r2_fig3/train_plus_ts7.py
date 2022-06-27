import mltrain as mlt

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad']


if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('ts7.xyz'),
                        box=None)

    ace = mlt.potentials.ACE('da',
                             system=system)

    ace.training_data = mlt.ConfigurationSet('da_al_ts5.npz')

    ace.al_train(method_name='orca',
                         temp=500,
                         max_active_time=1000,
                         fix_init_config=True)
    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(configuration=system.random_configuration(),
                                   mlp=ace,
                                   fs=500,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    # and compare, plotting a parity diagram and E_true, ∆E and ∆F
    trajectory.compare(ace, 'orca')
