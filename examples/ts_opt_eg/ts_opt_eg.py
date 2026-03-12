import os
import time
import mlptrain as mlt
from mlptrain.log import logger
from mlptrain.tools.ts_opt import mlip_dimer_method


if __name__ == '__main__':
    # === 1. SELECT INPUT CONFIG (TS GUESS) ===

    # e.g. endo DA vac wB97M TS
    input_config_fp = 'cis_endo_TS_wB97M.xyz'
    save_name = 'endo_DA_vac'
    react_coords = [(1, 12), (6, 11)]

    # === 2. SELECT MODEL(S) ===
    model_fpaths = ['./OFF_FT_42.model']

    # === 3. SELECT OPTIMISER SETTINGS ===
    fmax = 0.001  # default = 0.001
    displace_mag = 0.05  # default = 0.05
    reverse_displace = False
    override_existing = True

    model_names = [
        os.path.splitext(os.path.basename(model_fpath))[0]
        for model_fpath in model_fpaths
    ]
    logger.info(f'Testing models: {model_names}')
    # ====================================

    for model_fpath, model_name in zip(model_fpaths, model_names):
        system = mlt.System(box=None)
        model = mlt.potentials.MACE(
            model_name, system, model_fpath=model_fpath
        )

        # run dimer method with each model
        start_time = time.time()
        mlip_dimer_method(
            input_config_fp,
            model,
            react_coords,
            save_name,
            displace_mag=displace_mag,
            reverse_displace=reverse_displace,
            fmax=fmax,
            override_existing=override_existing,
        )
        logger.info(
            f'Ran {model_name} dimer method in {time.time() - start_time:.4f}s...'
        )
