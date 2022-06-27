import mltrain as mlt

mlt.Config.n_cores = 12
mlt.Config.orca_keywords = ['MP2', 'def2-TZVP', 'NOFROZENCORE']


if __name__ == '__main__':

    system = mlt.System(mlt.Molecule('ts_pbe0.xyz'), box=None)
    ace = mlt.potentials.ACE('da', system=system)

    #dft_al_data = mlt.ConfigurationSet()
    #dft_al_data.load_xyz('da_data.xyz', charge=0, mult=1)
    #dft_al_data.save('da_data.npz')

    # dft_al_data.single_point(method_name='orca')
    # dft_al_data.save('da_data.npz')

    ace.set_atomic_energies('orca')
    ace.train(mlt.ConfigurationSet('da_data.npz'))

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(configuration=system.random_configuration(),
                                   mlp=ace,
                                   fs=500,
                                   temp=300,
                                   dt=0.5,
                                   interval=10)

    # and compare, plotting a parity diagram and E_true, ∆E and ∆F
    trajectory.compare(ace, 'orca')
