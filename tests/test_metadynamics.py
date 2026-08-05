import os
from types import SimpleNamespace

import numpy as np
import mlptrain as mlt
import pytest
from ase.io.trajectory import Trajectory as ASETrajectory

from mlptrain.sampling import metadynamics as metad_module
from mlptrain.utils import work_in_tmp_dir
from .data.utils import work_in_zipped_dir

mlt.Config.n_cores = 2
here = os.path.abspath(os.path.dirname(__file__))


# --------------------------------------------------------------------------
# Timeout degradation
#
# These use synchronous stand-ins for the process pools so the None-handling
# paths can be reached without running (or timing out) real MD.
# --------------------------------------------------------------------------


class _FakePool:
    """Synchronous stand-in for multiprocessing.Pool."""

    def __init__(self, results):
        self._results = list(results)
        self._idx = 0

    def apply_async(self, func, args, kwds):
        result = self._results[self._idx]
        self._idx += 1

        return SimpleNamespace(get=lambda r=result: r)

    def close(self):
        return None

    def join(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_pool_context(results):
    return SimpleNamespace(Pool=lambda processes: _FakePool(results))


def _fake_executor(results):
    """Synchronous stand-in for concurrent.futures.ProcessPoolExecutor."""

    class _FakeExecutor:
        def __init__(self, max_workers=None, mp_context=None):
            self._idx = 0

        def submit(self, fn, *args, **kwargs):
            result = results[self._idx]
            self._idx += 1

            return SimpleNamespace(result=lambda r=result: r)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    return _FakeExecutor


@work_in_tmp_dir()
def test_estimate_width_raises_when_all_runs_time_out(
    h2_configuration, test_potential, monkeypatch
):
    """Every width run timing out is fatal, not a silently empty result.

    Previously ``np.min`` was called on an empty array and raised an opaque
    ValueError from deep inside numpy.
    """

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    monkeypatch.setattr(
        metad_module.mp, 'get_context', lambda method: _fake_pool_context([[]])
    )

    with pytest.raises(RuntimeError, match='Width estimation failed'):
        metad.estimate_width(
            configurations=h2_configuration,
            mlp=test_potential('1D'),
            plot=False,
            fs=100,
        )


@work_in_tmp_dir()
def test_estimate_width_uses_surviving_runs(
    h2_configuration, test_potential, monkeypatch
):
    """A timed-out run is dropped; the remaining widths still give a result.

    This also pins the ``all_widths = np.array(...)`` placement: building the
    array inside the collection loop turned the second iteration into a
    ragged append and broke ``np.min(axis=0)``.
    """

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    displaced = h2_configuration.copy()
    displaced.atoms[0].coord[0] += 0.1

    configurations = mlt.ConfigurationSet()
    configurations.append(h2_configuration)
    configurations.append(displaced)
    assert len(configurations) == 2

    monkeypatch.setattr(
        metad_module.mp,
        'get_context',
        lambda method: _fake_pool_context([[], [0.07]]),
    )

    widths = metad.estimate_width(
        configurations=configurations,
        mlp=test_potential('1D'),
        plot=False,
        fs=100,
    )

    assert widths == pytest.approx([0.07])


@work_in_tmp_dir()
def test_run_metadynamics_bails_out_when_all_runs_time_out(
    h2_configuration, test_potential, monkeypatch, mlp_caplog
):
    """All trajectories timing out degrades to a warning, not a crash.

    Previously the empty trajectory list was passed straight to
    ``_move_and_save_files`` and the post-processing raised.
    """

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    monkeypatch.setattr(
        metad_module, 'ProcessPoolExecutor', _fake_executor([None, None])
    )

    metad.run_metadynamics(
        configuration=h2_configuration,
        mlp=test_potential('1D'),
        n_runs=2,
        temp=300,
        dt=1,
        interval=10,
        pace=100,
        width=0.05,
        height=0.1,
        biasfactor=3,
        fs=100,
    )

    assert (
        'All metadynamics trajectories were skipped due to MD timeout.'
        in mlp_caplog.messages
    )
    assert not os.path.exists('trajectories')


@pytest.fixture
def run_metadynamics(test_potential):
    def _run_metadynamics(
        metad,
        n_runs,
        configuration,
        al_iter=None,
        save_sep=False,
        all_to_xyz=False,
        restart=False,
        **kwargs,
    ):
        metad.run_metadynamics(
            configuration=configuration,
            mlp=test_potential('1D'),
            temp=300,
            dt=1,
            interval=10,
            pace=100,
            width=0.05,
            height=0.1,
            biasfactor=3,
            al_iter=al_iter,
            n_runs=n_runs,
            save_sep=save_sep,
            all_to_xyz=all_to_xyz,
            restart=restart,
            **kwargs,
        )

    return _run_metadynamics


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_run_metadynamics(h2_configuration, run_metadynamics):
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    assert metad.bias is not None

    run_metadynamics(
        metad, n_runs, h2_configuration, all_to_xyz=True, save_fs=200, fs=500
    )

    assert os.path.exists('trajectories')
    assert os.path.exists('trajectories/combined_trajectory.xyz')

    metad_dir = 'plumed_files/metadynamics'
    for idx in range(1, n_runs + 1):
        assert os.path.exists(f'trajectories/trajectory_{idx}.traj')

        for sim_time in [200, 400]:
            assert os.path.exists(
                f'trajectories/' f'trajectory_{idx}_{sim_time}fs.traj'
            )
            assert os.path.exists(
                f'trajectories/' f'metad_{idx}_{sim_time}fs.xyz'
            )

        assert os.path.exists(os.path.join(metad_dir, f'colvar_cv1_{idx}.dat'))
        assert os.path.exists(os.path.join(metad_dir, f'HILLS_{idx}.dat'))

        assert os.path.exists(f'gaussian_heights/gaussian_heights_{idx}.pdf')

    metad.compute_fes(n_bins=100)

    for idx in range(1, n_runs + 1):
        assert os.path.exists(f'plumed_files/metadynamics/fes_{idx}.dat')

    assert os.path.exists('fes_raw.npy')
    fes_raw = np.load('fes_raw.npy')

    # 1 cv, 4 fes -> 5; 100 bins
    assert np.shape(fes_raw) == (5, 100)

    metad.plot_fes('fes_raw.npy')
    assert os.path.exists('metad_free_energy.pdf')

    metad.plot_fes_convergence(stride=2, n_surfaces=2)

    # 500 / 100: simulation time divided by the pace <=> number of gaussians
    # Surfaces are computed every 2 gaussians
    n_computed_surfaces = (500 / 100) // 2
    for idx in range(int(n_computed_surfaces)):
        assert os.path.exists(f'plumed_files/fes_convergence/fes_1_{idx}.dat')

    assert os.path.exists('fes_convergence/fes_convergence_diff.pdf')
    assert os.path.exists('fes_convergence/fes_convergence.pdf')


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_run_metadynamics_restart(h2_configuration, run_metadynamics):
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    run_metadynamics(metad, n_runs, h2_configuration, fs=500)

    run_metadynamics(metad, n_runs, h2_configuration, restart=True, fs=500)

    n_steps = len(
        np.loadtxt('plumed_files/metadynamics/colvar_cv1_1.dat', usecols=0)
    )
    n_gaussians = len(
        np.loadtxt('plumed_files/metadynamics/HILLS_1.dat', usecols=0)
    )

    # Adding two 500 fs simulations with interval 10 -> 51 frames each, but
    # removing one duplicate frame
    assert n_steps == 51 + 51 - 1
    assert n_gaussians == 5 + 5

    assert os.path.exists('trajectories/trajectory_1.traj')

    trajectory = ASETrajectory('trajectories/trajectory_1.traj')

    # Adding two 1 ps simulations with interval 10 -> 101 frames each, but
    # removing one duplicate frame (same as before, except testing this for
    # the generated .traj file instead of .dat file)
    assert len(trajectory) == 51 + 51 - 1


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_run_metadynamics_with_inherited_bias(
    h2_configuration, run_metadynamics
):
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    run_metadynamics(metad, n_runs, h2_configuration, al_iter=3, fs=500)

    run_metadynamics(
        metad, n_runs, h2_configuration, al_iter=3, restart=True, fs=500
    )

    metad_dir = 'plumed_files/metadynamics'
    for idx in range(1, n_runs + 1):
        assert os.path.exists(f'trajectories/trajectory_{idx}.traj')

        assert os.path.exists(os.path.join(metad_dir, f'colvar_cv1_{idx}.dat'))
        assert os.path.exists(os.path.join(metad_dir, f'HILLS_{idx}.dat'))

    metad.compute_fes(via_reweighting=True)
    assert os.path.exists('fes_raw.npy')


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_run_metadynamics_with_component(h2_configuration, run_metadynamics):
    cv1 = mlt.PlumedCustomCV('plumed_cv_dist.dat', 'x')
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    run_metadynamics(metad, n_runs, h2_configuration, fs=100)

    metad_dir = 'plumed_files/metadynamics'
    for idx in range(1, n_runs + 1):
        assert os.path.exists(
            os.path.join(metad_dir, f'colvar_cv1_x_{idx}.dat')
        )


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_run_metadynamics_with_additional_cvs(
    h2o_configuration, run_metadynamics
):
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    cv2 = mlt.PlumedAverageCV('cv2', (2, 1))
    cv2.attach_upper_wall(location=3.0, kappa=150.0)

    bias = mlt.PlumedBias(cvs=(cv1, cv2))

    metad = mlt.Metadynamics(cvs=cv1, bias=bias)

    assert metad.bias == bias
    assert metad.n_cvs == 1

    n_runs = 1
    run_metadynamics(
        metad,
        configuration=h2o_configuration,
        n_runs=n_runs,
        write_plumed_setup=True,
        fs=100,
    )

    with open('plumed_files/metadynamics/plumed_setup.dat', 'r') as f:
        plumed_setup = [line.strip() for line in f]

    # Not including the units
    assert plumed_setup[1:] == [
        'cv1_dist1: DISTANCE ATOMS=1,2',
        'cv1: CUSTOM ARG=cv1_dist1 VAR=cv1_dist1 '
        f'FUNC={1/1}*(cv1_dist1) PERIODIC=NO',
        'cv2_dist1: DISTANCE ATOMS=3,2',
        'cv2: CUSTOM ARG=cv2_dist1 VAR=cv2_dist1 '
        f'FUNC={1/1}*(cv2_dist1) PERIODIC=NO',
        'UPPER_WALLS ARG=cv2 AT=3.0 KAPPA=150.0 EXP=2',
        'metad: METAD ARG=cv1 PACE=100 HEIGHT=0.1 '
        'SIGMA=0.05 TEMP=300 BIASFACTOR=3 '
        'FILE=HILLS_1.dat',
        'PRINT ARG=cv1,cv1_dist1 ' 'FILE=colvar_cv1_1.dat STRIDE=10',
        'PRINT ARG=cv2,cv2_dist1 ' 'FILE=colvar_cv2_1.dat STRIDE=10',
    ]


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_estimate_width(h2_configuration, test_potential):
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    width = metad.estimate_width(
        configurations=h2_configuration,
        mlp=test_potential('1D'),
        plot=True,
        fs=100,
    )

    assert len(width) == 1

    files_directory = 'plumed_files/width_estimation'
    plots_directory = 'width_estimation'

    assert os.path.isdir(files_directory)
    assert os.path.exists(os.path.join(files_directory, 'colvar_cv1_1.dat'))

    assert os.path.isdir(plots_directory)
    assert os.path.exists(os.path.join(plots_directory, 'cv1_config1.pdf'))


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_try_multiple_biasfactors(h2_configuration, test_potential):
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)
    biasfactors = range(5, 11, 5)

    metad.try_multiple_biasfactors(
        configuration=h2_configuration,
        mlp=test_potential('1D'),
        temp=300,
        interval=10,
        dt=1,
        pace=100,
        width=0.05,
        height=0.1,
        biasfactors=biasfactors,
        plotted_cvs=cv1,
        fs=100,
    )

    files_dir = 'plumed_files/multiple_biasfactors'
    assert os.path.isdir(files_dir)

    plots_dir = 'multiple_biasfactors'
    assert os.path.isdir(plots_dir)

    for idx, biasf in enumerate(biasfactors, start=1):
        assert os.path.exists(os.path.join(files_dir, f'colvar_cv1_{idx}.dat'))
        assert os.path.exists(os.path.join(plots_dir, f'cv1_biasf{biasf}.pdf'))


@work_in_zipped_dir(os.path.join(here, 'data/data.zip'))
def test_block_analysis(h2_configuration, test_potential):
    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)
    dt = 1
    interval = 10
    n_runs = 1
    ps = 2
    start_time = 0.5

    metad.run_metadynamics(
        configuration=h2_configuration,
        mlp=test_potential('1D'),
        temp=300,
        dt=dt,
        interval=interval,
        pace=100,
        width=0.05,
        height=0.1,
        biasfactor=3,
        n_runs=n_runs,
        ps=ps,
    )

    metad.block_analysis(start_time=start_time)

    assert os.path.exists('block_analysis.pdf')
    assert os.path.exists('block_analysis.npz')

    start_time_fs = start_time * 1e3
    n_steps = int(start_time_fs / dt)
    n_used_frames = n_steps // interval

    min_n_blocks = 10
    min_blocksize = 10
    blocksize_interval = 10
    max_blocksize = n_used_frames // min_n_blocks

    data = np.load('block_analysis.npz')

    # axis 0: CV1; axis 1: 300 bins
    assert np.shape(data['CVs']) == (1, 300)
    for blocksize in range(
        min_blocksize, max_blocksize + 1, blocksize_interval
    ):
        # axis 0: error; axis 1: 300 bins
        assert np.shape(data[str(blocksize)]) == (3, 300)
