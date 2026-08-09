import mlptrain as mlt
from mlptrain.descriptor import SoapDescriptor

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = ['PBE0', 'def2-SVP', 'EnGrad']


if __name__ == '__main__':
    system = mlt.System(mlt.Molecule('da_ts.xyz'), box=None)
    mlp = mlt.potentials.MACE('da', system=system)

    descriptor = SoapDescriptor()
    mlp.al_train(
        method_name='orca',
        temp=300,  # K
        selection_method=mlt.selection.AtomicEnvSimilarity(descriptor),
        max_active_time=200,  # fs
        fix_init_config=True,
    )

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(
        configuration=system.configuration,
        mlp=mlp,
        fs=300,
        temp=100,
        dt=0.5,
        interval=10,
    )

    # and compare, plotting a parity plots and E_true, ∆E and ∆F
    trajectory.compare(mlp, 'orca')
