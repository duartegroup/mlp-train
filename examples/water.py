import mlptrain as mlt

mlt.Config.n_cores = 10


if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('water.xyz'), box=None)

    ace = mlt.potentials.ACE('water', system=system)
    ace.al_train(method_name='xtb', temp=500)

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(configuration=system.random_configuration(),
                                   mlp=ace,
                                   fs=200,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    trajectory.save(filename='water_trajectory.xyz')
