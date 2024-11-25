import mlptrain as mlt

mlt.Config.n_cores = 30
mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad', 'CPCM(Water)']

if __name__ == '__main__':
    # Initialise the system
    system = mlt.System(mlt.Molecule('r1_ts.xyz', charge=-1, mult=1), box=None)

    # Define CV and attach an upper wall
    avg_r = mlt.PlumedAverageCV(name='avg_r', atom_groups=((0, 1), (0, 2)))
    avg_r.attach_upper_wall(location=2.5, kappa=1000)

    # Define CVs for extra information
    r_f = mlt.PlumedAverageCV(name='r_f', atom_groups=(0, 1))
    r_cl = mlt.PlumedAverageCV(name='r_cl', atom_groups=(0, 2))

    # Initialise PlumedBias
    bias = mlt.PlumedBias(cvs=(avg_r, r_f, r_cl))

    # Define the potential and train using Downhill AL (fix_init_config=True)
    ace = mlt.potentials.ACE('r1_downhill', system=system)
    ace.al_train(
        method_name='orca',
        temp=500,
        n_configs_iter=5,
        max_active_iters=50,
        min_active_iters=20,
        fix_init_config=True,
        bias=bias,
    )
