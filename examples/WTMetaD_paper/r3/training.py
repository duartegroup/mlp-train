import mlptrain as mlt
from ase.constraints import FixAtoms
from autode.atoms import Atom
import numpy as np

mlt.Config.n_cores = 30
mlt.Config.orca_keywords = [
    'PBE',
    'D3BJ',
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
    solvation = mlt.Configuration(
        atoms=autode_atoms,
        charge=solute_config.charge + solvent_config.charge,
        mult=1,
        box=solvent_config.box,
    )
    return solvation


def generate_init_configs(
    n, solute_xyz, solvent_xyz, box_size=[18.5, 18.5, 18.5]
):
    solute = mlt.ConfigurationSet()
    solute.load_xyz(
        filename=solute_xyz, charge=1, mult=1, box=mlt.box.Box(box_size)
    )
    solute = solute[0]

    solvent_mol = mlt.Molecule(name=solvent_xyz, charge=0, mult=1)
    solvent_system = mlt.System(solvent_mol, box=mlt.box.Box(box_size))
    solvent_system.add_molecules(solvent_mol, num=58)

    init_configs = mlt.ConfigurationSet()

    while len(init_configs) < n:
        solvent = solvent_system.random_configuration()
        solvent.box = mlt.box.Box(box_size)
        solvated = solvation(
            solute_config=solute, solvent_config=solvent, apm=5, radius=1.7
        )
        if solvated.n_atoms == 274:
            init_configs.append(solvated)

    return init_configs


if __name__ == '__main__':
    r_1 = mlt.PlumedAverageCV(name='r_1', atom_groups=(4, 41))
    r_1.attach_upper_wall(location=5.5, kappa=1000)
    r_2 = mlt.PlumedAverageCV(name='r_2', atom_groups=(4, 34))
    r_2.attach_upper_wall(location=5.5, kappa=1000)

    # Define CV for WTMetaD AL (r_1 - r_2)
    diff_r = mlt.PlumedDifferenceCV(
        name='diff_r', atom_groups=((4, 41), (4, 34))
    )

    bias = mlt.PlumedBias(cvs=(r_1, r_2, diff_r))
    bias.initialise_for_metad_al(width=0.05, cvs=diff_r, biasfactor=70)

    selector = mlt.selection.AtomicEnvSimilarity(threshold=0.9996)

    system_gas = mlt.System(
        mlt.Molecule('gly_gas.xyz', charge=1, mult=1), box=None
    )
    ace_gas = mlt.potentials.ACE('gly_gas_ace', system=system_gas)
    ace_gas.al_train(
        method_name='orca',
        selection_method=selector,
        temp=300,
        n_init_configs=5,
        n_configs_iter=5,
        max_active_iters=50,
        min_active_iters=30,
        inherit_metad_bias=True,
        bias=bias,
    )

    mol_CCl2 = mlt.Molecule('CCl2.xyz', charge=0, mult=1)
    system_sol = mlt.System(mol_CCl2, box=mlt.box.Box([14.5, 14.5, 14.5]))
    system_sol.add_molecules(mol_CCl2, num=27)
    # Define the potential
    ace_sol = mlt.potentials.ACE('CCl2_ace', system=system_sol)
    ace_sol.al_train(
        method_name='orca',
        temp=300,
        selection_method=selector,
        max_active_iters=50,
        min_active_iters=10,
        max_active_time=5000,
        pbc=True,
        box_size=[14.6, 14.6, 14.6],
    )

    init_configs = generate_init_configs(5, 'gly_gas.xyz', 'CCl2.xyz')

    system = mlt.System(
        mlt.Molecule('gly_gas.xyz', charge=1, mult=1),
        box=mlt.box.Box([18.5, 18.5, 18.5]),
    )
    system.add_molecules(mol_CCl2, num=44)

    ace_ex = mlt.potentials.ACE('gly_explicit_ace', system=system)

    fix_reactant = [
        i for i in range(mlt.Molecule('gly_gas.xyz', charge=1, mult=1).n_atoms)
    ]
    ace_ex.al_train(
        method_name='orca',
        temp=300,
        selection_method=selector,
        init_configs=init_configs,
        fix_init_config=False,
        max_active_iters=6,
        min_active_iters=3,
        max_active_time=1000,
        pbc=True,
        box_size=[18.6, 18.6, 18.6],
        constraints=FixAtoms(fix_reactant),
    )

    ace_ex.al_train(
        method_name='orca',
        temp=300,
        selection_method=selector,
        n_configs_iter=5,
        max_active_iters=50,
        min_active_iters=10,
        max_active_time=3000,
        inherit_metad_bias=True,
        pbc=True,
        box_size=[18.6, 18.6, 18.6],
        bias=bias,
        bias_start_iter=0,
    )

    ace = mlt.potentials.ACE('gly_ace', system=system)

    ace.training_data += ace_gas.training_data
    ace.training_data += ace_sol.training_data
    ace.training_data += ace_ex.training_data

    ace.set_atomic_energies(method_name='orca')
    ace.train()
