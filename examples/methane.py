import mltrain as mlt

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['PBE', 'def2-SVP', 'EnGrad']


if __name__ == '__main__':

    # Set up the system of a methane molecule without any periodic boundaries
    system = mlt.System(mlt.Molecule('methane.xyz'),
                        box=None)

    # Initialise a Gaussian Approximation Potential for this system
    gap = mlt.potentials.GAP('methane',
                             system=system)

    # and train using active learning at 1000 K
    gap.al_train(method_name='orca', temp=1000)

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(configuration=system.random_configuration(),
                                   mlp=gap,
                                   fs=200,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    # and compare, plotting a parity diagram and E_true, ∆E and ∆F
    trajectory.compare('orca')
