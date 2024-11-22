import autode as ade
import mlptrain as mlt

mlt.Config.n_cores = 20
mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad', 'CPCM(Water)']

if __name__ == '__main__':
    
    # Initialise the system
    system = mlt.System(mlt.Molecule('ch3cl_f.xyz', charge=-1, mult=1),
                        box=None)

    # Define CV and attach an upper wall
    avg_r = mlt.PlumedAverageCV(name='avg_r', atom_groups=((0, 1), (0, 2)))
    avg_r.attach_upper_wall(location=2.5, kappa=1000)

    # Define CVs for extra information
    r_f = mlt.PlumedAverageCV(name='r_f', atom_groups=(0, 1))
    r_cl = mlt.PlumedAverageCV(name='r_cl', atom_groups=(0, 2))

    # Define CV for WTMetaD AL (r_cl - r_f)
    diff_r = mlt.PlumedDifferenceCV(name='diff_r', atom_groups=((0, 2), (0, 1)))

    # Initialise PlumedBias for WTMetaD AL
    bias = mlt.PlumedBias(cvs=(avg_r, r_f, r_cl, diff_r))
    bias.initialise_for_metad_al(width=0.05,
                                 cvs=diff_r,
                                 biasfactor=100)

    # Define the potential and train using WTMetaD AL (inherit_metad_bias=True)
    ace = mlt.potentials.ACE('r1_wtmetad', system=system)
    ace.al_train(method_name='orca',
                 temp=300,
                 n_init_configs=5,
                 n_configs_iter=5,
                 max_active_iters=50,
                 min_active_iters=30,
                 inherit_metad_bias=True,
                 bias=bias)
