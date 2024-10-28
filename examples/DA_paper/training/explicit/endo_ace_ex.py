import mlptrain as mlt
from mlptrain.box import Box
from mlptrain.training.selection import AtomicEnvSimilarity
from mlptrain.configurations.explicit_solvation import generate_init_configs, sample_randomly_from_configset


mlt.Config.n_cores = 10
mlt.Config.orca_keywords = [
    'wB97M-D3BJ',
    'def2-TZVP',
    'def2/J',
    'RIJCOSX',
    'EnGrad',
]

if __name__ == '__main__':
    water_mol = mlt.Molecule(name='h2o.xyz')
    ts_mol = mlt.Molecule(name='cis_endo_TS_wB97M.xyz')

    # generate sub training set of pure water system by AL training
    water_system = mlt.System(water_mol, box=Box([100, 100, 100]))
    water_system.add_molecules(water_mol, num=26)
    Water_mlp = mlt.potentials.ACE('water_sys', water_system)
    water_init = generate_init_configs(n=10, bulk_water=True, TS=False)
    Water_mlp.al_train(
        method_name='orca',
        selection_method=AtomicEnvSimilarity(),
        fix_init_config=True,
        init_configs=water_init,
        max_active_time=5000,
    )

    # generate sub training set of TS in water system by AL training
    ts_in_water = mlt.System(ts_mol, box=Box([100, 100, 100]))
    ts_in_water.add_molecules(water_mol, num=40)
    ts_in_water_mlp = mlt.potentials.ACE('TS_in_water', ts_in_water)
    ts_in_water_init = generate_init_configs(n=10, bulk_water=True, TS=True)
    ts_in_water_mlp.al_train(
        method_name='orca',
        selection_method=AtomicEnvSimilarity(),
        fix_init_config=True,
        init_configs=ts_in_water_init,
        max_active_time=5000,
    )

    # generate sub training set of TS with two water system by AL training
    ts_2water = mlt.System(ts_mol, box=Box([100, 100, 100]))
    ts_2water.add_molecules(water_mol, num=2)
    ts_2water_mlp = mlt.potentials.ACE('TS_2water', ts_2water)
    ts_2water_init = generate_init_configs(n=10, bulk_water=False, TS=True)
    ts_2water_mlp.al_train(
        method_name='orca',
        selection_method=AtomicEnvSimilarity(),
        fix_init_config=True,
        init_configs=ts_2water_init,
        max_active_time=5000,
    )

    # generate sub training set of TS in gas phase by AL training
    ts_gasphase = mlt.System(ts_mol, box=Box([100, 100, 100]))
    ts_gasphase_mlp = mlt.potentials.ACE('TS_gasphase', ts_gasphase)
    ts_gasphase_mlp.al_train(
        method_name='orca',
        selection_method=AtomicEnvSimilarity(),
        fix_init_config=True,
        max_active_time=5000,
    )

    # combined sub training set to get the finally potential
    system = mlt.System(ts_mol, box=Box([100, 100, 100]))
    system.add_molecules(water_mol, num=40)
    endo = mlt.potentials.ACE('endo_in_water_ace_wB97M', system)
    pure_water_config = sample_randomly_from_configset(
        Water_mlp.training_data, 50
    )
    ts_in_water_config = sample_randomly_from_configset(
        ts_in_water_mlp.training_data, 250
    )
    ts_2water_config = sample_randomly_from_configset(
        ts_2water_mlp.training_data, 150
    )
    gasphase_config = sample_randomly_from_configset(
        ts_gasphase_mlp.training_data, 150
    )

    endo.training_data += pure_water_config
    endo.training_data += ts_in_water_config
    endo.training_data += ts_2water_config
    endo.training_data += gasphase_config

    endo.set_atomic_energies(method_name='orca')
    endo.train()
