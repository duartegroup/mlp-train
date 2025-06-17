from mlptrain.analysis.calc_pes import calculate_pes
import pytest
import os
import mlptrain as mlt
import autode as ade
from mlptrain.utils import work_in_tmp_dir
import torch

N_CORES = 1
mlt.Config.n_cores = N_CORES
ade.Config.n_cores = N_CORES

if torch.cuda.is_available():
    mlt.Config.mace_params['calc_device'] = 'cuda'


@pytest.fixture
def diels_alder_config_in_vac():
    return


@pytest.fixture
def diels_alder_config_in_water():
    return


@work_in_tmp_dir()
def test_mace_vac_diels_alder_pes():
    """Test PES of Diels Alder system in vacuum with MACE model."""

    # input parameters
    model_name = 'MACE-MP0'
    ts_xyz_fpath = 'cis_endo_TS_wB97M.xyz'
    save_name = 'cis_endo_DA_vac'
    react_coords = [(1, 12), (6, 11)]
    opt_fmax = 0.3
    grid_spec = (2.0, 2.1, 2)

    # load model
    system = mlt.System(box=[100.0, 100.0, 100.0])
    cwd = os.getcwd()
    mlp = mlt.potentials.MACE(
        model_name, system, model_fpath=f'{cwd}/{model_name}.model'
    )

    calculate_pes(
        mlp,
        ts_xyz_fpath,
        react_coords,
        save_name,
        opt_fmax=opt_fmax,
        grid_spec=grid_spec,
    )

    assert True


@work_in_tmp_dir()
def test_mace_water_diels_alder_pes():
    """Test PES of Diels Alder system in water with MACE model"""

    # input parameters
    model_name = 'MACE-MP0'
    ts_xyz_fpath = 'cis_endo_TS_wB97M.xyz'
    save_name = 'cis_endo_DA_water'
    react_coords = [(1, 12), (6, 11)]
    opt_fmax = 0.3
    grid_spec = (2.0, 2.1, 2)

    # solvent parameters
    solvent_xyz_fpath = 'h2o.xyz'
    solvation_box_size = 14.0
    solvent_density = 0.99657

    # load model
    system = mlt.System(box=[100.0, 100.0, 100.0])
    cwd = os.getcwd()
    mlp = mlt.potentials.MACE(
        model_name, system, model_fpath=f'{cwd}/{model_name}.model'
    )

    calculate_pes(
        mlp,
        ts_xyz_fpath,
        react_coords,
        save_name,
        solvent_xyz_fpath=solvent_xyz_fpath,
        solvent_density=solvent_density,
        solvation_box_size=solvation_box_size,
        opt_fmax=opt_fmax,
        grid_spec=grid_spec,
    )

    assert True
