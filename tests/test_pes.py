from mlptrain.analysis.calc_pes import calculate_pes
from tests.data.utils import work_in_zipped_dir
import pytest
import os
import mlptrain as mlt
import autode as ade
import torch

N_CORES = 1
mlt.Config.n_cores = N_CORES
ade.Config.n_cores = N_CORES

here = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def diels_alder_config_in_vac():
    configs = mlt.ConfigurationSet()

    with open('tmp.xyz', 'w') as xyz_file:
        print(
            '22',
            'Coordinates from ORCA-job cis_endo_TS',
            'C   -2.48551827153476      4.22046776301455     -0.76387019964855',
            'C   -3.29706997853341      3.20800825403778      0.00560550135848',
            'C   -2.40613152285159      2.20188143774424      0.39280166083415',
            'H   -4.17193907669498      3.47732454004277      0.57922874323701',
            'C   -1.25421626341582      2.28073881423334     -0.39134018674370',
            'H   -2.62971697790748      1.40772166988209      1.08790738827116',
            'C   -1.38578121697094      3.33888426662266     -1.27292693453425',
            'H   -0.46531442710123      1.54577034081137     -0.40594438977404',
            'H   -0.63060343945869      3.66681988004509     -1.97069031398589',
            'H   -2.05245199670347      4.92740600649740     -0.04650437268625',
            'H   -3.02666202824448      4.78295774452280     -1.51796301673495',
            'C   -3.06504740042196      2.35793555636474     -2.60937979146807',
            'C   -4.11128678715421      2.39158047287693     -1.69752866089065',
            'H   -4.87265150897932      3.15276283427014     -1.79773583587097',
            'H   -4.44050572009143      1.44897075672146     -1.28720724310498',
            'C   -2.34020015940562      1.10203379410996     -2.81792271712678',
            'H   -2.93852127963818      3.14151880876616     -3.34300331696286',
            'O   -2.50386109363881      0.13973783317053     -2.08940214569736',
            'C   -1.36442140742721      1.05457064210129     -3.97140892173567',
            'H   -0.69600040010072      1.91533933637728     -3.94662341331746',
            'H   -1.91361709328836      1.09471264172471     -4.91346276230566',
            'H   -0.79009611473740      0.13380188786275     -3.92976894681267',
            sep='\n',
            file=xyz_file,
        )

    configs.load_xyz('tmp.xyz', charge=0, mult=1)
    return configs[0]


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_mace_vac_diels_alder_pes(diels_alder_config_in_vac):
    """Test PES of Diels Alder system in vacuum with MACE model."""

    # TODO: add MACE_model in data/data dir + transitions state

    if torch.cuda.is_available():
        mlt.Config.mace_params['calc_device'] = 'cuda'

    # input params
    model_name = 'MACE-MP0'
    save_name = 'cis_endo_DA'
    react_coords = [(1, 12), (6, 11)]
    opt_fmax = 0.5
    box_dim = [100.0, 100.0, 100.0]
    grid_spec = (2.0, 2.2, 2)

    # load model
    system = mlt.System(box=box_dim)
    cwd = os.getcwd()
    mlp = mlt.potentials.MACE(
        model_name, system, model_fpath=f'{cwd}/{model_name}.model'
    )

    # load ts config from xyz
    ts_config = diels_alder_config_in_vac

    calculate_pes(
        mlp,
        ts_config,
        react_coords,
        save_name,
        opt_fmax=opt_fmax,
        grid_spec=grid_spec,
        box_dim=box_dim,
        kept_file_exts=None,
    )
