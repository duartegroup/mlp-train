import mlptrain as mlt

mlt.Config.n_cores = 8

if __name__ == '__main__':

    # Initialise the system

    h2o_system = mlt.System(mlt.Molecule('h2o.xyz'), box=None)

    # Generate a starting metadynamics configuration

    h2o_config = h2o_system.random_configuration()
 
    # Define CVs, can also attach walls to them. More complicated CVs (i.e.
    # not DISTANCE, ANGLE, or TORSION; e.g. PATH) can be defined using
    # PlumedCustomCV that requires a PLUMED-like input file containing the
    # definition of the CV

    cv1 = mlt.PlumedAverageCV(name='cv1', atom_groups=((0, 1), (0, 2)))
    cv1.attach_upper_wall(location=5, kappa=10000)

    cv2 = mlt.PlumedAverageCV(name='cv2', atom_groups=(1, 0, 2))
    cv2.attach_upper_wall(location=3.14, kappa=10000)

    # Attach CVs to the metadynamics object

    metad = mlt.Metadynamics(cvs=(cv1, cv2))

    # Load (or train) a machine learning potential

    ace = mlt.potentials.ACE(name='water', system=h2o_system)

    # Can run optional methods (estimate_width() and try_multiple_biafactors()) 
    # to help choose appropriate metadynamics parameters (width and bias factor),

    width = metad.estimate_width(configurations=h2o_config,
                                 mlp=ace,
                                 plot=True)

    metad.try_multiple_biasfactors(configuration=h2o_config,
                                   mlp=ace,
                                   temp=300,
                                   interval=10,
                                   dt=1,
                                   width=width,
                                   biasfactors=(5, 10, 15),
                                   plotted_cvs=(cv1, cv2),
                                   ps=40)

    # Execute metadynamics production runs, 8 independent simulations are
    # performed in parallel

    metad.run_metadynamics(configuration=h2o_config,
                           mlp=ace,
                           temp=300,
                           interval=10,
                           dt=1,
                           width=width,
                           biasfactor=5,
                           n_runs=8,
                           restart=False,
                           ps=40)

    # Plot the resulting free energy surface (FES), the same method can be used
    # to plot the FES from block analysis or FES from a previous simulation
    # using .npy file, for all the options see the documentation of plot_fes()

    metad.plot_fes()

    # Can run optional methods (plot_fes_convergence() and block_analysis())
    # to estimate the convergence of the production simulation

    metad.plot_fes_convergence(stride=10, n_surfaces=5)

    # Block analysis method also generates a set of free energy surfaces with
    # different block sizes which can be used in plot_fes() method

    metad.block_analysis()
