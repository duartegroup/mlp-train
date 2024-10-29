import numpy as np
from typing import List, Tuple
from autode.atoms import Atom
import mlptrain as mlt
from mlptrain.log import logger
from mlptrain.box import Box

def from_ase_to_autode(atoms):
    """
    Convert ase atoms to autode atoms.

    -----------------------------------------------------------------------
    Arguments:
        atoms: input ase atoms
    """
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


def add_water(solute: mlt.Configuration, n: int = 2):
    """
    Add water molecules to the reactive species.

    -----------------------------------------------------------------------
    Arguments:
        solute: mlt.Configuration, the molecule to add water molecules, should including box
        n: number of water molecules to add
    """
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


def solvation(solute_config: mlt.Configuration,
              solvent_config: mlt.Configuration,
              apm: int,
              radius: float,
              enforce: bool = True) -> mlt.Configuration:
    """
    Function to generate solvated system by adding the solute at the center of box,
    then remove the overlapped solvent molecules (adapted from https://doi.org/10.1002/qua.26343)

    -----------------------------------------------------------------------
    Arguments:
        solute: mlt.Configuration() solute.box is not None
        solvent: mlt.Configuration() solvent.box is not None
        apm: number of atoms per solvent molecule
        radius: cutout radius around each solute atom
        enforce: True / False Wrap solvent regardless of previous solvent PBC choices
    Returns:

    """
    assert solute_config.box is not None, 'configuration must have box'
    assert solvent_config.box is not None, 'configuration must have box'

    solute = solute_config.ase_atoms
    solvent = solvent_config.ase_atoms

    def wrap(D, cell, pbc):
        """
        Wrap distance to nearest neighbor

        -----------------------------------------------------------------------
        Arguments:
            D: distance
        """
        for i, periodic in enumerate(pbc):
            if periodic:
                d = D[:, i]
                L = cell[i]
                d[:] = (d + L / 2) % L - L / 2
        return None

    def molwrap(atoms, n, idx=0):
        """Wrap to cell without breaking molecule.

        -----------------------------------------------------------------------
        Arguments:
            n: number of atoms per solvent molecule
            idx: which atom in the solvent molecule to determine molecular distances from
        """
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


def generate_init_solv_configs(n: int,
                               solvent_mol: mlt.Molecule,
                               bulk_solvent: bool = True,
                               include_TS: bool = True,
                               TS_info: Tuple[str, int, int] = None) -> mlt.ConfigurationSet:
    """
    Generate initial configuration to train potential.
    It can generate three sets (pure solvent, TS immersed in solvent and TS bounded two solvent molecules)
    of initial configuration by modifying the boolean variables:

    Three possible options:
    1. bulk_solvent = True, include_TS = False --> bulk solvent only
    2. bulk_solvent = True, include_TS = True  --> TS immersed in explicit solvent
    3. bulk_solvent = False, include_TS = True --> TS bounded to two solvent molecules only

    -----------------------------------------------------------------------
    Arguments:
        n: number of init_configs
        solvent_mol: mlt.Molecule object for explicit solvent molecule to add
        bulk_solvent: whether to include a solution
        include_TS: whether to include the TS of the reaction in the system
        TS_info: tuple of (file path, charge, spin multiplicity) for TS configuration
    Returns:
        (mlt.ConfigurationSet): initial set of configurations
    """

    # load TS in box
    TS: mlt.Configuration = None
    if include_TS:
        assert TS_info is not None, "If include TS is set to true, a TS file must be provided..."

        TS_file_path, TS_charge, TS_mult = TS_info
        TS_config_set = mlt.ConfigurationSet()
        TS_config_set.load_xyz(filename=TS_file_path, charge=TS_charge, mult=TS_mult)
        TS = TS_config_set[0]
        TS.box = Box([11, 11, 11])
        TS.charge = TS_charge
        TS.mult = TS_mult

    init_configs = mlt.ConfigurationSet()

    if bulk_solvent:

        # TS immersed in a solvent box
        if include_TS:
            solvent_system = mlt.System(solvent_mol, box=Box([11, 11, 11]))
            solvent_system.add_molecules(solvent_mol, num=43)
            for i in range(n):
                solvated = solvation(
                    solute_config=TS,
                    solvent_config=solvent_system.random_configuration(),
                    apm=3,
                    radius=1.7,
                )
                init_configs.append(solvated)

        # pure solvent box
        else:
            solvent_system = mlt.System(solvent_mol, box=Box([9.32, 9.32, 9.32]))
            solvent_system.add_molecules(solvent_mol, num=26)

            for i in range(n):
                pure_water = solvent_system.random_configuration()
                init_configs.append(pure_water)

    # TS bounded with two solvent molecules (in case of water and DA reaction, at carbonyl group to form hydrogen bond)
    else:
        assert include_TS, "If bulk_solvent is false, the TS must be included..."
        for i in range(n):
            TS_with_water = add_water(solute=TS, n=2)
            init_configs.append(TS_with_water)

    # Change the box of system to extremely large to imitate cluster system
    # the box is needed for ACE potential
    for config in init_configs:
        config.box = Box([100, 100, 100])

    return init_configs


def sample_randomly_from_configset(configurationset: mlt.ConfigurationSet, size: int) -> List[mlt.Configuration]:
    """
    Simply returns a randomly sampled sub-set of configurations from the
    input ConfigurationSet with a given size.

    -----------------------------------------------------------------------
    Arguments:
        configurationset: the set of configurations to sample
        size: the number of samples to choose
    Returns:
        (List[Configuration]): randomly sampled subset of input configuration set
    """
    return list(np.random.choice(configurationset, size=size))