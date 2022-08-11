import mlptrain as mlt

mlt.Config.n_cores = 10


if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('sn2.xyz', charge=-1), box=None)

    ace = mlt.potentials.ACE('sn2', system=system)
    ace.al_train(method_name='xtb', temp=500, fix_init_config=True)
