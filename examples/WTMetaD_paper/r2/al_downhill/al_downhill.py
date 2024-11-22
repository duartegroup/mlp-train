import autode as ade
import mlptrain as mlt

# ORCA PATH (change accordingly)
ade.Config.ORCA.path = '/usr/local/orca_5_0_3/orca'

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['PBE0', 'D3BJ', 'def2-SVP', 'EnGrad']

if __name__ == '__main__':

    # Initialise the system
    system = mlt.System(mlt.Molecule('ts_r2.xyz', charge=0, mult=1),
                        box=None)

    # Define CVs for extra information
    avg_r = mlt.PlumedAverageCV(name='avg_r', atom_groups=((14, 11), (14, 10)))
    r_1 = mlt.PlumedAverageCV(name='r_1', atom_groups=(14, 11))
    r_2 = mlt.PlumedAverageCV(name='r_2', atom_groups=(14, 10))
    diff_r = mlt.PlumedDifferenceCV(name='diff_r', atom_groups=((14, 11), (14, 10)))

    # Initialise PlumedBias
    bias = mlt.PlumedBias(cvs=(avg_r, r_1, r_2, diff_r))

    # Define the potential and train using Downhill AL (fix_init_config=True)
    ace = mlt.potentials.ACE('r2_downhill', system=system)
    ace.al_train(method_name='orca',
                 temp=500,
                 n_init_configs=10,
                 n_configs_iter=10,
                 max_active_iters=50,
                 min_active_iters=20,
                 fix_init_config=True,
                 bias=bias)
