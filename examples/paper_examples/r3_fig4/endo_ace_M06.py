import mltrain as mlt

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['M062X', 'RIJCOSX', 'def2/J', 'def2-SVP', 'EnGrad']

if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('cis_endo_TS_M06.xyz', charge=0, mult=1),
                        box=None)

    ace = mlt.potentials.ACE('endo_ace', system=system)

    ace.al_train(method_name='orca',
                 temp=500,
                 max_active_time=1000,
                 fix_init_config=True)

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(configuration=system.configuration,
                                   mlp=ace,
                                   fs=200,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    # and compare, plotting a parity diagram and E_true, ∆E and ∆F
    trajectory.compare(ace, 'orca')
