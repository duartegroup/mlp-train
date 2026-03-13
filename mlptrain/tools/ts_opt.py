"""
Dimer method to find optimal TS using MLIP.
"""
import os
import numpy as np
from ase.io import write as ase_write
from ase.io import read as ase_read
from ase import Atoms
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
import mlptrain as mlt
from mlptrain.log import logger
from mlptrain.loss import RMSE
from mlptrain.potentials._base import MLPotential


def mlip_dimer_method(
    input_config_fp: str,
    model: MLPotential,
    react_coords: list[tuple],
    save_name: str,
    ref_ts_fp: str = None,
    out_dir: str = '.',
    displace_mag: float = 0.05,
    reverse_displace: bool = False,
    fmax: float = 0.001,
    override_existing: bool = False,
) -> Atoms:
    """
    Runs dimer TS Optimisation method with an MLIP.

    Parameters:
        input_config_fp (str): Path to input configuration file
        model_ (MLPotential): MLIP model to run TS Opt with.
        react_coords (list[tuple]): List of reaction coordinate tuples (bonds forming / breaking) [(a1 -> a2), (a3 -> a4)].
        save_name (str): Name for saving output files.
        ref_ts_fp (Optional(str): Path to a reference TS for comparison to print output statistics.
        out_dir (Optional(str)): Output directory. Defaults to OPT_DIR.
        displace_mag (float, optional): Magnitude of initial atomic displacement along reaction coordinate for dimer. Defaults to 0.05.
        reverse_displace (float, optional): Whether to reverse the displace direction (in some cases if atoms are not displaced towards eachother use this setting.)
        fmax (float, optional): Optimisation convergence threshold. Defaults to 0.001.
        override_existing: whether to override previous runs of the same name

    Returms:
        ase.Atoms: optimised TS configuration as an ASE Atoms object.
    """

    # check for existing opt ts with this save name, and if not overriding existing run, skip
    opt_ts_save_fp = f'{out_dir}/{save_name}_{model.name}_opt_TS.xyz'
    if not override_existing and os.path.exists(opt_ts_save_fp):
        logger.info(f'Detected OPT TS file, {opt_ts_save_fp} skipping...')
        ts_config = (
            mlt.Configuration().from_xyz(filename=opt_ts_save_fp).ase_atoms
        )
        return ts_config

    # load input config
    input_config = (
        mlt.Configuration().from_xyz(filename=input_config_fp).ase_atoms
    )

    input_config.calc = model.ase_calculator
    input_config.get_potential_energy()
    input_xyz = input_config.get_positions()

    # Set up the dimer
    react_coord_ids = set(
        [a_id for r_coord in react_coords for a_id in r_coord]
    )
    logger.info(f'React Coord IDs: {react_coord_ids}')
    with DimerControl(
        initial_eigenmode_method='displacement',
        displacement_method='vector',
        maximum_translation=0.1,
        logfile=None,
        mask=[i in react_coord_ids for i in range(len(input_config))],
    ) as d_control:
        d_atoms = MinModeAtoms(input_config, d_control)

        # displace the atoms along reaction coordinate
        rev_fac = 1.0 if reverse_displace else -1.0
        displacement_vector = [[0.0] * 3 for _ in range(len(input_config))]

        for i, j in react_coords:
            di = input_xyz[j] - input_xyz[i]
            dj = input_xyz[i] - input_xyz[j]
            di = di / np.linalg.norm(di)
            dj = dj / np.linalg.norm(dj)
            displacement_vector[i] = list(displace_mag * rev_fac * di)
            displacement_vector[j] = list(displace_mag * rev_fac * dj)
        d_atoms.displace(displacement_vector=displacement_vector)

        # converge to a saddle point
        traj_save_fp = f'{out_dir}/{save_name}_{model.name}_dimer_trj.trj'
        with MinModeTranslate(
            d_atoms,
            trajectory=traj_save_fp,
            logfile=f'{out_dir}/{save_name}_{model.name}_dimer_opt_log.txt',
        ) as dim_rlx:
            logger.info(
                f'Running dimer optimisation on {input_config_fp} with {model.name} (fmax={fmax})...'
            )
            dim_rlx.run(fmax=fmax)

        traj_configs = ase_read(traj_save_fp, index=':')

        # save full trajectory
        old_traj_fp = traj_save_fp
        ase_write(
            traj_save_fp.replace('.trj', '.xyz'), traj_configs, format='extxyz'
        )
        os.remove(old_traj_fp)
        logger.info(
            f"Saved optimisation trajectory to: {traj_save_fp.replace('.trj', '.xyz')}"
        )

        # save optimised TS (last frame)
        opt_ts_config = traj_configs[-1]
        ase_write(opt_ts_save_fp, opt_ts_config, format='extxyz')
        logger.info(f'Saved Opt TS to: {opt_ts_save_fp}')

        opt_TS_r_dists = [
            opt_ts_config.get_distance(*r_atoms, mic=True)
            for r_atoms in react_coords
        ]
        logger.info(
            'MLIP Opt TS R Coord Dists:'
            + ' '.join(
                [
                    f' r{i} ({r_coord}): {opt_TS_r_dists[i]:.3f}'
                    for i, r_coord in enumerate(react_coords)
                ]
            )
        )

        # print stats w.r.t reference if specified
        if ref_ts_fp is not None:
            ref_ts_config = (
                mlt.Configuration().from_xyz(filename=ref_ts_fp).ase_atoms
            )
            ref_r_dists = [
                ref_ts_config.get_distance(*r_atoms, mic=True)
                for r_atoms in react_coords
            ]
            logger.info('Reference TS fpath found, using for comparison...')
            logger.info(
                'Ref. TS Distances:'
                + ' '.join(
                    [
                        f' r{i} ({r_coord}): {ref_r_dists[i]:.3f}'
                        for i, r_coord in enumerate(react_coords)
                    ]
                )
            )
            react_coord_dist_error = sum(
                [
                    abs(ref_r_dists[i] - opt_TS_r_dists[i])
                    for i, _ in enumerate(react_coords)
                ]
            )
            logger.info(
                f'React Coord Dist. Error: {react_coord_dist_error:.3f}'
            )
            struct_rmse = RMSE.statistic(
                opt_ts_config.get_positions() - ref_ts_config.get_positions()
            )
            logger.info(
                f'MLIP OPT TS Reactants Structural RMSD: {struct_rmse:.3f}'
            )

    # return optimised TS
    return opt_ts_config
