import mlptrain as mlt

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad']


if __name__ == '__main__':
    system = mlt.System(mlt.Molecule('ts_pbe0.xyz'), box=None)
    gap = mlt.potentials.GAP('da', system=system)

    gap.al_train(
        method_name='orca',
        temp=300,  # K
        selection_method=mlt.selection.MaxAtomicEnvDistance(),
        max_active_time=200,  # fs
        fix_init_config=True,
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
