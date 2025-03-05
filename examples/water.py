import mlptrain as mlt
from mlptrain.training.selection import AtomicEnvSimilarity
from mlptrain.descriptor import SoapDescriptor

mlt.Config.n_cores = 10

if __name__ == '__main__':
    system = mlt.System(mlt.Molecule('water.xyz'), box=None)

    ace = mlt.potentials.ACE('water', system=system)

    SoapDescriptor = SoapDescriptor(
        average='outer', r_cut=6.0, n_max=6, l_max=6
    )
    selector = AtomicEnvSimilarity(descriptor=descriptor, threshold=0.9995)
    ace.al_train(method_name='xtb', selection_method=selector, temp=500)

    # Run some dynamics with the potential
    trajectory = mlt.md.run_mlp_md(
        configuration=system.random_configuration(),
        mlp=ace,
        fs=200,
        temp=300,
        dt=0.5,
        interval=10,
    )

    trajectory.save(filename='water_trajectory.xyz')
