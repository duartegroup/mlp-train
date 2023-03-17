import os
import time
import numpy as np
import mlptrain as mlt
from .test_potential import TestPotential
from .utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


def _h2_umbrella():
    return mlt.UmbrellaSampling(zeta_func=mlt.AverageDistance([0, 1]), kappa=100)


def _h2_pulled_traj():
    traj = mlt.ConfigurationSet()
    traj.load_xyz(os.path.join(here, 'data', 'h2_traj.xyz'), charge=0, mult=1)

    return traj


def _h2_sparse_traj():
    traj = _h2_pulled_traj()
    sparse_traj = mlt.ConfigurationSet()
    sparse_traj.append(traj[0])
    sparse_traj.append(traj[-1])

    return sparse_traj


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_umbrella():

    umbrella = _h2_umbrella()
    traj = _h2_pulled_traj()
    n_windows = 3

    assert umbrella.kappa is not None and np.isclose(umbrella.kappa, 100.)
    assert umbrella.zeta_refs is None

    # Zeta refs are now reset
    umbrella.run_umbrella_sampling(traj,
                                   mlp=TestPotential('1D'),
                                   temp=300,
                                   interval=5,
                                   dt=0.5,
                                   n_windows=n_windows,
                                   save_sep=False,
                                   all_to_xyz=True,
                                   fs=1000,
                                   save_fs=300)

    # Sampling with a high force constant should lead to fitted Gaussians
    # that closely match the reference (target) values
    for window in umbrella.windows:
        assert window.gaussian_plotted is not None
        assert np.isclose(window.gaussian_plotted.mean, window.zeta_ref, atol=0.1)

    assert os.path.exists('trajectories')
    assert os.path.exists('trajectories/combined_trajectory.xyz')

    for idx in range(1, n_windows + 1):
        assert os.path.exists(f'trajectories/trajectory_{idx}.traj')

        for sim_time in [300, 600, 900]:
            assert os.path.exists(f'trajectories/trajectory_{idx}_{sim_time}fs.traj')
            assert os.path.exists(f'trajectories/window_{idx}_{sim_time}fs.xyz')

    assert os.path.exists('fitted_data.pdf')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_umbrella_parallel():

    execution_time = {}

    for n_cores in (1, 4):

        mlt.Config.n_cores = n_cores

        umbrella = _h2_umbrella()
        traj = _h2_pulled_traj()

        start = time.perf_counter()
        umbrella.run_umbrella_sampling(traj,
                                       mlp=TestPotential('1D'),
                                       temp=300,
                                       interval=5,
                                       dt=0.5,
                                       n_windows=4,
                                       fs=500)
        finish = time.perf_counter()

        execution_time[n_cores] = finish - start

    # Calculation with more cores should run faster
    assert execution_time[4] < execution_time[1]


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_umbrella_sparse_traj():

    umbrella = _h2_umbrella()
    traj = _h2_sparse_traj()
    n_windows = 9

    # Indices from 1 to 9
    zeta_refs = umbrella._reference_values(traj=traj,
                                           num=n_windows,
                                           final_ref=None,
                                           init_ref=None)

    middle_ref = zeta_refs[5]
    middle_bias = mlt.Bias(zeta_func=umbrella.zeta_func,
                           kappa=umbrella.kappa,
                           reference=middle_ref)

    # There should be no good starting frame for the middle window (index 5)
    # as the sparse trajectory only contains the initial and final frame
    assert umbrella._no_ok_frame_in(traj, middle_ref)

    umbrella.run_umbrella_sampling(traj,
                                   mlp=TestPotential('1D'),
                                   temp=300,
                                   interval=5,
                                   dt=0.5,
                                   n_windows=n_windows,
                                   fs=100,
                                   save_sep=True)

    assert os.path.exists('trajectories')
    assert os.path.isdir('trajectories')

    previous_window_traj = mlt.ConfigurationSet()
    previous_window_traj.load_xyz(filename='trajectories/window_4.xyz',
                                  charge=0,
                                  mult=1)

    middle_window_traj = mlt.ConfigurationSet()
    middle_window_traj.load_xyz(filename='trajectories/window_5.xyz',
                                charge=0,
                                mult=1)

    closest_frame = umbrella._best_init_frame(bias=middle_bias,
                                              traj=previous_window_traj)
    starting_frame = middle_window_traj[0]

    # The starting frame for the middle window (index 5) should be
    # the closest frame from the previous window (index 4)
    assert starting_frame == closest_frame


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_umbrella_save_load():

    umbrella = _h2_umbrella()
    traj = _h2_pulled_traj()

    umbrella.run_umbrella_sampling(traj,
                                   mlp=TestPotential('1D'),
                                   temp=300,
                                   interval=5,
                                   dt=0.5,
                                   n_windows=3,
                                   fs=100,
                                   save_sep=False)

    umbrella.save(folder_name='tmp_us')
    assert os.path.exists('tmp_us') and os.path.isdir('tmp_us')

    loaded = mlt.UmbrellaSampling.from_folder(folder_name='tmp_us', temp=300)
    assert len(loaded.windows) == 3
    assert np.allclose(loaded.zeta_refs, umbrella.zeta_refs)

    for idx, window in enumerate(loaded.windows):
        assert np.isclose(window.zeta_ref, umbrella.zeta_refs[idx])
        assert np.isclose(window._bias.kappa, 100)
        assert len(window._obs_zetas) == 41
