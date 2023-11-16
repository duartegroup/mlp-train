import mlptrain as mlt

mlt.Config.n_cores = 4

if __name__ == '__main__':
    # Define a reaction coordinate as R1 - R2
    umbrella = mlt.UmbrellaSampling(
        zeta_func=mlt.DifferenceDistance((0, 1), (0, 5)), kappa=20
    )

    irc = mlt.ConfigurationSet()
    irc.load_xyz(filename='irc.xyz', charge=-1, mult=1)

    system = mlt.System(mlt.Molecule('sn2.xyz', charge=-1, mult=1), box=None)

    # Run umbrella sampling across the IRC using GAP MD
    umbrella.run_umbrella_sampling(
        irc,
        mlp=mlt.potentials.GAP('sn2', system=system),
        temp=300,
        interval=5,
        dt=0.5,
        n_windows=10,
        ps=1,
    )

    # Use WHAM to calculate the free energy
    umbrella.wham()
