import mlptrain as mlt

mlt.Config.n_cores = 1


if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('water.xyz'), box=None)

    mace = mlt.potentials.MACE('water', system=system)
    
    mace.al_train(method_name='xtb', temp=500, max_active_iters=2, max_active_time=50, n_configs_iter=3, md_program="OpenMM")

    # Run some dynamics with the potential
    trajectory = mlt.md_openmm.run_mlp_md_openmm(configuration=system.random_configuration(),
                                   mlp=mace,
                                   fs=2000,
                                   temp=300,
                                   dt=0.5,
                                   interval=100)

    trajectory.save(filename='water_trajectory.xyz')
