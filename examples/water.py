import mlptrain as mlt

# NOTE: This example assumes that you have xTB installed
# conda install xtb-python

N_CORES = 8
mlt.Config.n_cores = N_CORES


if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('water.xyz'), box=None)

    ace = mlt.potentials.ACE('water', system=system)
    ace.al_train(method_name='xtb', temp=500, n_configs_iter=N_CORES)

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(configuration=system.random_configuration(),
                                   mlp=ace,
                                   fs=200,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    trajectory.save(filename='water_trajectory.xyz')
