import mlptrain as mlt

mlt.Config.n_cores = 10

if __name__ == '__main__':
    system = mlt.System(mlt.Molecule('r_r2.xyz', charge=0, mult=1), box=None)

    # Load reactant configuration
    config = mlt.ConfigurationSet()
    config.load_xyz('r_r2.xyz', charge=0, mult=1)
    config = config[0]

    # Define CVs for extra information
    r_1 = mlt.PlumedAverageCV(name='r_1', atom_groups=(14, 11))
    r_2 = mlt.PlumedAverageCV(name='r_2', atom_groups=(14, 10))

    # Define CV for WTMetaD (r_1 - r_2)
    diff_r = mlt.PlumedDifferenceCV(
        name='diff_r', atom_groups=((14, 11), (14, 10))
    )

    # Load potential
    ace = mlt.potentials.ACE('r2_wtmetad', system=system)

    # Initialise PlumedBias
    bias = mlt.PlumedBias(cvs=(r_1, r_2, diff_r))

    # Initialise Metadynamics object
    metad = mlt.Metadynamics(cvs=diff_r, bias=bias)

    # Perform MLP well-tempered metadynamics
    # (load inherited bias with index 15)
    metad.run_metadynamics(
        configuration=config,
        mlp=ace,
        temp=365.6,
        interval=10,
        dt=0.5,
        pace=200,
        biasfactor=80,
        n_runs=10,
        al_iter=15,
        ps=250,
    )

    # Compute and plot FES (via reweighting)
    metad.compute_fes(
        n_bins=500,
        via_reweighting=True,
        bandwidth=0.02,
        temp=365.6,
        dt=0.5,
        interval=10,
    )
    metad.plot_fes()

    # Perform block analysis
    metad.block_analysis(
        start_time=0, temp=365.6, dt=0.5, interval=10, min_n_blocks=30
    )
