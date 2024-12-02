import mlptrain as mlt

mlt.Config.n_cores = 15

if __name__ == '__main__':
    system = mlt.System(mlt.Molecule('ts_r2.xyz', charge=0, mult=1), box=None)

    # Load IRC
    irc = mlt.ConfigurationSet()
    irc.load_xyz('r2_irc.xyz', charge=0, mult=1)

    # Load potential
    mlp = mlt.potentials.ACE('r2_wtmetad', system=system)

    # Define CV for umbrella sampling (r_1 - r_2)
    zeta_func = mlt.DifferenceDistance((14, 11), (14, 10))

    # Define UmbrellaSampling object
    umbrella = mlt.UmbrellaSampling(zeta_func=zeta_func, kappa=20, temp=365.6)

    # Perform MLP umbrella sampling
    umbrella.run_umbrella_sampling(
        traj=irc, mlp=mlp, temp=365.6, interval=10, dt=0.5, n_windows=30, ps=40
    )

    # Truncate first 10 ps from each window
    umbrella.truncate_window_trajectories(removed_fraction=0.25)

    # Compute free energy using WHAM
    umbrella.wham(n_bins=500)
