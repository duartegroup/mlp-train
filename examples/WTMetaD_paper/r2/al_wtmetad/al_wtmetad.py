import mlptrain as mlt

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['PBE0', 'D3BJ', 'def2-SVP', 'EnGrad']

if __name__ == '__main__':
    system = mlt.System(mlt.Molecule('r_r2.xyz', charge=0, mult=1), box=None)

    # Define CV and attach an upper wall
    avg_r = mlt.PlumedAverageCV(name='avg_r', atom_groups=((14, 11), (14, 10)))
    avg_r.attach_upper_wall(location=2.5, kappa=1000)

    # Define CVs for extra information
    r_1 = mlt.PlumedAverageCV(name='r_1', atom_groups=(14, 11))
    r_2 = mlt.PlumedAverageCV(name='r_2', atom_groups=(14, 10))

    # Define CV for WTMetaD AL (r_1 - r_2)
    diff_r = mlt.PlumedDifferenceCV(
        name='diff_r', atom_groups=((14, 11), (14, 10))
    )

    # Initialise PlumedBias for WTMetaD AL
    bias = mlt.PlumedBias(cvs=(avg_r, r_1, r_2, diff_r))
    bias.initialise_for_metad_al(width=0.05, cvs=diff_r, biasfactor=90)

    # Define the potential and train using WTMetaD AL (inherit_metad_bias=True)
    ace = mlt.potentials.ACE('isoindene_2_metad_2', system=system)
    ace.al_train(
        method_name='orca',
        temp=300,
        n_init_configs=5,
        n_configs_iter=10,
        max_active_iters=40,
        min_active_iters=20,
        inherit_metad_bias=True,
        bias=bias,
    )
