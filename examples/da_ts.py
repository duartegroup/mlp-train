import mlptrain as mlt

mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad']


if __name__ == '__main__':
    system = mlt.System(mlt.Molecule('da_ts.xyz'), box=None)
    gap = mlt.potentials.GAP('da', system=system)

    gap.al_train(
        method_name='xtb',
        temp=300,  # K
        selection_method=mlt.selection.AtomicEnvSimilarity(),
        max_active_time=200,  # fs
        fix_init_config=True,
        n_configs_iter=1,
    )

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(
        configuration=system.configuration,
        mlp=gap,
        fs=300,
        temp=100,
        dt=0.5,
        interval=10,
    )

    # and compare, plotting a parity plots and E_true, ∆E and ∆F
    trajectory.compare(gap, 'orca')
