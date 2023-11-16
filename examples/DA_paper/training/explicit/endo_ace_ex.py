import mlptrain as mlt
import numpy as np
from autode.atoms import Atom
from mlptrain.log import logger
from mlptrain.box import Box
from mlptrain.training.selection import MaxAtomicEnvDistance

mlt.Config.n_cores = 10
mlt.Config.orca_keywords = [
    'wB97M-D3BJ',
    'def2-TZVP',
    'def2/J',
    'RIJCOSX',
    'EnGrad',
]


def from_ase_to_autode(atoms):
    # atoms is ase.Atoms
    autode_atoms = []
    symbols = atoms.symbols

    for i in range(len(atoms)):
        autode_atoms.append(
            Atom(
                symbols[i],
                x=atoms.positions[i][0],
                y=atoms.positions[i][1],
                z=atoms.positions[i][2],
            )
        )

    return autode_atoms


def add_water(solute, n=2):
    """add water molecules to the reactive species
    solute: mlt.Configuration, the molecule to add water molecules, should including box
    n: number of water molecules to add"""
    from ase import Atoms
    from ase.calculators.tip3p import rOH, angleHOH

    # water molecule
    x = angleHOH * np.pi / 180 / 2
    pos = [
        [0, 0, 0],
        [0, rOH * np.cos(x), rOH * np.sin(x)],
        [0, rOH * np.cos(x), -rOH * np.sin(x)],
    ]
    water = Atoms('OH2', positions=pos)
    H_origin = water[0].position - water[1].position
    water.translate(H_origin)

    water0 = water.copy()
    water1 = water.copy()
    water0.rotate(180, 'x')
    water0.rotate(180, 'z')

    assert solute.box is not None, 'configuration must have box'
    sol = solute.ase_atoms
    sol.center()
    sys = sol.copy()

    # randomly rotate water molecule
    water0.rotate(
        np.random.uniform(0, 180),
        (0, np.random.uniform(-1, 0), np.random.uniform(0, 1)),
    )
    sys += water0
    water1.rotate(
        np.random.uniform(0, 180),
        (np.random.uniform(-1, 0), np.random.uniform(0, 1), 0),
    )
    if n >= 2:
        for i in range(n - 1):
            sys += water1

    len_sol = len(sol)
    sol_idx = list(range(len_sol))
    idx = list(range(len(sys)))

    C_idx = []
    O_idx = []
    for atm in sol_idx:
        if sys.numbers[atm] == 6:
            C_idx.append(atm)
        if sys.numbers[atm] == 8:
            O_idx.append(atm)

    # the direction to add water molecules to avioding unphysical cases, system specific
    C98 = (sys[C_idx[7]].position - sys[C_idx[8]].position) / np.linalg.norm(
        sys[C_idx[7]].position - sys[C_idx[8]].position
    )
    C68 = (sys[C_idx[7]].position - sys[C_idx[5]].position) / np.linalg.norm(
        sys[C_idx[7]].position - sys[C_idx[5]].position
    )
    C48 = (sys[C_idx[7]].position - sys[C_idx[3]].position) / np.linalg.norm(
        sys[C_idx[7]].position - sys[C_idx[3]].position
    )
    C8O = (sys[O_idx[0]].position - sys[C_idx[7]].position) / np.linalg.norm(
        sys[O_idx[0]].position - sys[C_idx[7]].position
    )
    direction = [C68, C48, C8O, C98]
    water_idx = []

    for atm in idx[22::3]:
        single_water = []
        for i in range(3):
            single_water.append(atm + i)
        water_idx.append(single_water)
    assert len(water_idx) == n

    for j in range(len(water_idx)):
        displacement = np.random.uniform(1.85, 2.4)
        logger.info(
            f'distance between H in water and O is TS is {displacement} '
        )
        vec = displacement * direction[j]
        water = water_idx[j]
        trans = sys[O_idx[0]].position + vec
        for mol in water_idx[j]:
            sys[mol].position += trans

    autode_atoms = from_ase_to_autode(atoms=sys)
    added_water = mlt.Configuration(atoms=autode_atoms, box=solute.box)
    return added_water


def solvation(solute_config, solvent_config, apm, radius, enforce=True):
    """function to generate solvated system by adding the solute at the center of box,
    then remove the overlapped solvent molecules
    adapted from https://doi.org/10.1002/qua.26343
    solute: mlt.Configuration() solute.box is not None
    solvent: mlt.Configuration() solvent.box is not None
    aps: number of atoms per solvent molecule
    radius: cutout radius around each solute atom
    enforce: True / False Wrap solvent regardless of previous solvent PBC choices"""
    assert solute_config.box is not None, 'configuration must have box'
    assert solvent_config.box is not None, 'configuration must have box'

    solute = solute_config.ase_atoms
    solvent = solvent_config.ase_atoms

    def wrap(D, cell, pbc):
        """wrap distance to nearest neighbor
        D: distance"""
        for i, periodic in enumerate(pbc):
            if periodic:
                d = D[:, i]
                L = cell[i]
                d[:] = (d + L / 2) % L - L / 2
        return None

    def molwrap(atoms, n, idx=0):
        """Wrap to cell without breaking molecule
        n: number of atoms per solvent molecule
        idx: which atom in the solvent molecule to determine molecular distances from"""
        center = atoms.cell.diagonal() / 2
        positions = atoms.positions.reshape((-1, n, 3))
        distances = positions[:, idx] - center
        old_distances = distances.copy()
        wrap(distances, atoms.cell.diagonal(), atoms.pbc)
        offsets = distances - old_distances
        positions += offsets[:, None]
        atoms.set_positions(positions.reshape((-1, 3)))
        return atoms

    assert not (
        solvent.cell.diagonal() == 0
    ).any(), 'solvent atoms have no cell'
    assert (
        solvent.cell == np.diag(solvent.cell.diagonal())
    ).all(), 'sol cell not orthorhombic'
    if enforce:
        solvent.pbc = True
    sol = molwrap(solvent, apm)

    # put the solute ay the center of the solvent box
    solute.set_cell(sol.cell)
    solute.center()

    sys = solute + sol
    sys.pbc = True

    solute_idx = range(len(solute))
    mask = np.zeros(len(sys), bool)
    mask[solute_idx] = True

    # delete solvent molecules for whose atom is overlap with solute
    atoms_to_delete = []
    for atm in solute_idx:
        mol_dists = sys[atm].position - sys[~mask][::].positions
        idx = np.where((np.linalg.norm(mol_dists, axis=1)) < radius)
        for i in idx:
            list_idx = i.tolist()
        for i in list_idx:
            n = i % apm
            for j in range(apm - n):
                atoms_to_delete.append(i + j + len(solute_idx))
            for j in range(n + 1):
                atoms_to_delete.append(i - j + len(solute_idx))

    atoms_to_delete = np.unique(atoms_to_delete)
    del sys[atoms_to_delete]

    # conver ase atom to autode atom then to mlt configuation
    autode_atoms = from_ase_to_autode(atoms=sys)
    solvation = mlt.Configuration(atoms=autode_atoms)
    return solvation


def generate_init_configs(n, bulk_water=True, TS=True):
    """generate initial configuration to train potential
    it can generate three sets (pure water, TS immersed in water and TS bounded two water molecules)
    of initial configuration by modify the boolean variables
    n: number of init_configs
    bulk_water: whether to include a solution
    TS: whether to include the TS of the reaction in the system"""
    init_configs = mlt.ConfigurationSet()
    TS = mlt.ConfigurationSet()
    TS.load_xyz(filename='cis_endo_TS_wB97M.xyz')
    TS = TS[0]
    TS.box = Box([11, 11, 11])
    TS.charge = 0
    TS.mult = 1

    if bulk_water:
        # TS immersed in a water box
        if TS:
            water_mol = mlt.Molecule(name='h2o.xyz')
            water_system = mlt.System(water_mol, box=Box([11, 11, 11]))
            water_system.add_molecules(water_mol, num=43)
            for i in range(n):
                solvated = solvation(
                    solute_config=TS,
                    solvent_config=water_system.random_configuration(),
                    apm=3,
                    radius=1.7,
                )
                init_configs.append(solvated)

        # pure water box
        else:
            water_mol = mlt.Molecule(name='h2o.xyz')
            water_system = mlt.System(water_mol, box=Box([9.32, 9.32, 9.32]))
            water_system.add_molecules(water_mol, num=26)

            for i in range(n):
                pure_water = water_system.random_configuration()
                init_configs.append(pure_water)

    # TS bounded with two water molecules at carbonyl group to form hydrogen bond
    else:
        assert TS is True, 'cannot generate initial configuration'
        for i in range(n):
            TS_with_water = add_water(solute=TS, n=2)
            init_configs.append(TS_with_water)

    # Change the box of system to extermely large to imitate cluster system
    # the box is needed for ACE potential
    for config in init_configs:
        config.box = Box([100, 100, 100])
    return init_configs


def remove_randomly_from_configset(configurationset, remainder):
    configSet = list(np.random.choice(configurationset, size=remainder))
    return configSet


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
        selection_method=MaxAtomicEnvDistance(),
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
        selection_method=MaxAtomicEnvDistance(),
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
        selection_method=MaxAtomicEnvDistance(),
        fix_init_config=True,
        init_configs=ts_2water_init,
        max_active_time=5000,
    )

    # generate sub training set of TS in gas phase by AL training
    ts_gasphase = mlt.System(ts_mol, box=Box([100, 100, 100]))
    ts_gasphase_mlp = mlt.potentials.ACE('TS_gasphase', ts_gasphase)
    ts_gasphase_mlp.al_train(
        method_name='orca',
        selection_method=MaxAtomicEnvDistance(),
        fix_init_config=True,
        max_active_time=5000,
    )

    # combined sub training set to get the finally potential
    system = mlt.System(ts_mol, box=Box([100, 100, 100]))
    system.add_molecules(water_mol, num=40)
    endo = mlt.potentials.ACE('endo_in_water_ace_wB97M', system)
    pure_water_config = remove_randomly_from_configset(
        Water_mlp.training_data, 50
    )
    ts_in_water_config = remove_randomly_from_configset(
        ts_in_water_mlp.training_data, 250
    )
    ts_2water_config = remove_randomly_from_configset(
        ts_2water_mlp.training_data, 150
    )
    gasphase_config = remove_randomly_from_configset(
        ts_gasphase_mlp.training_data, 150
    )

    endo.training_data += pure_water_config
    endo.training_data += ts_in_water_config
    endo.training_data += ts_2water_config
    endo.training_data += gasphase_config

    endo.set_atomic_energies(method_name='orca')
    endo.train()
