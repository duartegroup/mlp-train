import mlptrain as mlt

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad']


if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('ts_pbe0.xyz'),
                        box=None)

    mlp = mlt.potentials.NeQUIP('da', system=system)

    mlp.al_train(method_name='orca',
                         temp=500, 
                         max_active_time=500,
                         fix_init_config=True)

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(configuration=system.random_configuration(),
                                   mlp=mlp,
                                   fs=500,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    # and compare, plotting a parity diagram and E_true, ∆E and ∆F
    trajectory.compare(mlp, 'orca')
