import os
import numpy as np
import mltrain as mlt
from .test_potential import TestPotential
from .utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


def _h2_umbrella():
    return mlt.UmbrellaSampling(zeta_func=mlt.AverageDistance([0, 1]), kappa=100)


def _h2_pulled_traj():
    traj = mlt.ConfigurationSet()
    traj.load_xyz(os.path.join(here, 'data', 'h2_traj.xyz'), charge=0, mult=1)

    return traj


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_window_umbrella():

    umbrella = _h2_umbrella()
    traj = _h2_pulled_traj()

    assert umbrella.kappa is not None and np.isclose(umbrella.kappa, 100.)
    assert umbrella.zeta_refs is None

    # Setting the reference values of the reaction coordinate should un-None
    umbrella._set_reference_values(traj, num=3, init_ref=0.8, final_ref=1.2)

    assert umbrella.zeta_refs is not None
    assert np.allclose(umbrella.zeta_refs, np.linspace(0.8, 1.2, 3))

    # Zeta refs are now reset
    umbrella.run_umbrella_sampling(traj,
                                   mlp=TestPotential('1D'),
                                   temp=300,
                                   interval=5,
                                   dt=0.5,
                                   n_windows=3,
                                   fs=1000)

    # Sampling with a high force constant should lead to fitted Gaussians
    # that closely match the reference (target) values
    for gaussian, ref in zip(umbrella._fitted_gaussians, umbrella.zeta_refs):
        assert np.isclose(gaussian.params[1], ref, atol=0.1)

    assert os.path.exists('combined_windows.xyz')
    assert os.path.exists('fitted_data.pdf')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_umbrella_save_load():

    umbrella = mlt.UmbrellaSampling(zeta_func=mlt.AverageDistance([0, 1]),
                                    kappa=100)

    traj = mlt.ConfigurationSet()
    traj.load_xyz('h2_traj.xyz', charge=0, mult=1)

    umbrella.run_umbrella_sampling(traj,
                                   mlp=TestPotential('1D'),
                                   temp=300,
                                   interval=5,
                                   dt=0.5,
                                   n_windows=3,
                                   fs=100)

    umbrella.save(folder_name='tmp_us')
    assert os.path.exists('tmp_us') and os.path.isdir('tmp_us')

    loaded = mlt.UmbrellaSampling.from_folder(folder_name='tmp_us')
    assert len(loaded.windows) == 3
    assert np.allclose(loaded.zeta_refs, umbrella.zeta_refs)

    for idx, window in enumerate(loaded.windows):
        assert np.isclose(window.zeta_ref, umbrella.zeta_refs[idx])
        assert np.isclose(window._bias.kappa, 100)
        assert len(window._obs_zetas) == 41
