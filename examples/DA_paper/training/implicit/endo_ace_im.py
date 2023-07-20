import mlptrain as mlt
from mlptrain.training.selection import MaxAtomicEnvDistance

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['wB97M-D3BJ', 'def2-TZVP','def2/J', 'RIJCOSX','EnGrad', 'CPCM(water)']

if __name__ == '__main__':
    mol_TS = mlt.Molecule('cis_endo_TS_water.xyz')

    TS = mlt.Configuration (charge = 0, mult =1 )
    TS.atoms += mol_TS.atoms.copy()

    system = mlt.System(mol_TS,
                        box=None)

    ace = mlt.potentials.ACE('endo_ace_wB97M_imwater',
                             system=system)

    selector = MaxAtomicEnvDistance()
    ace.al_train(method_name='orca',
                 selection_method = selector,
                 max_active_time = 5000,
                 fix_init_config = True,
                 temp=300)

    trajectory = mlt.md.run_mlp_md(configuration=TS,
                                   mlp=ace,
                                   fs=200,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    trajectory.compare(ace, 'orca')
