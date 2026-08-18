import json
import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from autode.atoms import Atom

from mlptrain.config import Config
from mlptrain.configurations import Configuration, ConfigurationSet
from mlptrain.training import active
from mlptrain.training.selection import AbsDiffE
from mlptrain.utils import work_in_tmp_dir


# These tests exercise the active-learning process plumbing and the HILLS
# hardening added in jl/timeout-resolution, without running expensive MD,
# MACE training, ORCA, or PLUMED. The bugs they protect against live in
# process/result handling and in file parsing, so the unit tests use cheap
# fake workers/stub subprocesses and a single spawned subprocess covers the
# real multiprocessing path.


def _config(x: float, energy: float = -1.0) -> Configuration:
    """Build a small, serialisable configuration for active-learning tests."""
    config = Configuration(atoms=[Atom('H', x, 0.0, 0.0)])
    config.energy.true = energy
    return config


# --------------------------------------------------------------------------
# Fake process plumbing
# --------------------------------------------------------------------------


class _FakeProcess:
    """Synchronous stand-in for multiprocessing.Process.

    Keeps the unit tests deterministic while still exercising the parent
    polling loop's expected process API: ``start()``, ``is_alive()``,
    ``join()``, ``terminate()``, ``kill()``, ``pid``, and ``exitcode``.

    ``hang=True`` starts a worker that never finishes, and ``stubborn=True``
    additionally makes it ignore ``terminate()`` so the SIGKILL escalation
    path is reached. The real spawn path is covered separately by
    ``test_add_active_configs_spawn_integration``.
    """

    _next_pid = 1000

    def __init__(self, target, args, hang=False, stubborn=False):
        self._target = target
        self._args = args
        self._hang = hang
        self._stubborn = stubborn
        self.pid = _FakeProcess._next_pid
        self.exitcode = None
        self.calls = []
        self._alive = False
        _FakeProcess._next_pid += 1

    def start(self):
        self._alive = True

        if self._hang:
            return

        try:
            self._target(*self._args)
        finally:
            self.exitcode = 0
            self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        return None

    def terminate(self):
        self.calls.append('terminate')

        if self._stubborn:
            return

        self.exitcode = -15
        self._alive = False

    def kill(self):
        self.calls.append('kill')
        self.exitcode = -9
        self._alive = False


class _FakeContext:
    """Small replacement for ``mp.get_context('spawn')`` in unit tests."""

    def __init__(self, hang=False, stubborn=False):
        self.hang = hang
        self.stubborn = stubborn
        self.processes = []

    def Queue(self):
        return Queue()

    def Process(self, target, args):
        process = _FakeProcess(
            target=target, args=args, hang=self.hang, stubborn=self.stubborn
        )
        self.processes.append(process)
        return process


class _FakeClock:
    """Stand-in for the ``time`` module with a fast-forwarding clock.

    Used to reach the poll loop's hard duration cap without waiting hours.
    """

    def __init__(self, step):
        self._t = 0.0
        self._step = step

    def monotonic(self):
        self._t += self._step
        return self._t

    def sleep(self, _):
        return None


def _install_fake_context(monkeypatch, hang=False, stubborn=False):
    ctx = _FakeContext(hang=hang, stubborn=stubborn)
    monkeypatch.setattr(active.mp, 'get_context', lambda method: ctx)
    return ctx


def _run_add_active_configs(mlp, n_configs, **kwargs):
    kwargs.setdefault('method_name', 'mock')
    kwargs.setdefault('iteration', 0)
    kwargs.setdefault('inherit_metad_bias', False)
    kwargs.setdefault('bias_start_iter', 0)
    kwargs.setdefault('max_time', 1.0)

    return active._add_active_configs(
        mlp,
        init_config=_config(99.0),
        selection_method=AbsDiffE(),
        n_configs=n_configs,
        **kwargs,
    )


# --------------------------------------------------------------------------
# _gen_active_config_worker
# --------------------------------------------------------------------------


def test_worker_puts_result_on_queue(monkeypatch):
    """A successful worker reports (idx, 'ok', config, None)."""

    returned = _config(1.0)
    monkeypatch.setattr(
        active, '_gen_active_config', lambda *args, **kwargs: returned
    )

    result_queue = Queue()
    active._gen_active_config_worker(
        result_queue,
        3,
        _config(0.0),
        mlp=None,  # ty: ignore[invalid-argument-type]
        selector=AbsDiffE(),
        n_cores=1,
        kwargs={'max_time': 1.0, 'method_name': 'mock'},
    )

    idx, status, config, err = result_queue.get_nowait()
    assert (idx, status, err) == (3, 'ok', None)
    assert config == returned


def test_worker_reports_exceptions_instead_of_raising(monkeypatch):
    """An exception inside the worker must not escape the entrypoint.

    The parent has no traceback across the process boundary, so the worker
    serialises the error onto the queue and exits cleanly instead.
    """

    def _boom(*args, **kwargs):
        raise RuntimeError('selection blew up')

    monkeypatch.setattr(active, '_gen_active_config', _boom)

    result_queue = Queue()
    active._gen_active_config_worker(
        result_queue,
        0,
        _config(0.0),
        mlp=None,  # ty: ignore[invalid-argument-type]
        selector=AbsDiffE(),
        n_cores=1,
        kwargs={'max_time': 1.0, 'method_name': 'mock'},
    )

    idx, status, config, err = result_queue.get_nowait()
    assert (idx, status, config) == (0, 'error', None)
    assert 'selection blew up' in err


# --------------------------------------------------------------------------
# _add_active_configs parent poll loop
# --------------------------------------------------------------------------


@work_in_tmp_dir()
def test_add_active_configs_collects_all_results(
    mlp_caplog, monkeypatch, test_potential
):
    """Happy path: every worker returns a configuration and all are kept."""

    results = {idx: _config(float(idx)) for idx in range(3)}
    monkeypatch.setattr(
        active,
        '_gen_active_config',
        lambda *args, **kwargs: results[kwargs['idx']],
    )
    _install_fake_context(monkeypatch)
    monkeypatch.setattr(Config, 'n_cores', 3)
    monkeypatch.setattr(Config, 'process_timeout', 30)

    mlp = test_potential()

    _run_add_active_configs(mlp, n_configs=3)

    assert mlp.n_train == 3
    assert 'Collected 3/3 active configurations' in mlp_caplog.messages
    assert os.path.exists('datasets/dataset_after_iter_0.npz')


@work_in_tmp_dir()
def test_add_active_configs_tolerates_failed_worker(
    mlp_caplog, monkeypatch, test_potential
):
    """A worker returning None is reported but does not abort the iteration."""

    results = {0: _config(0.0), 1: None, 2: _config(2.0)}
    monkeypatch.setattr(
        active,
        '_gen_active_config',
        lambda *args, **kwargs: results[kwargs['idx']],
    )
    _install_fake_context(monkeypatch)
    monkeypatch.setattr(Config, 'n_cores', 3)
    monkeypatch.setattr(Config, 'process_timeout', 30)

    mlp = test_potential()

    _run_add_active_configs(mlp, n_configs=3)

    assert mlp.n_train == 2
    assert (
        '1/3 active learning workers failed or timed out'
        in mlp_caplog.messages
    )
    assert 'Collected 2/3 active configurations' in mlp_caplog.messages


@work_in_tmp_dir()
def test_add_active_configs_survives_worker_exception(
    mlp_caplog, monkeypatch, test_potential
):
    """An exception drained off the queue is logged, not re-raised."""

    def _gen(*args, **kwargs):
        if kwargs['idx'] == 0:
            raise ValueError('bad frame')

        return _config(float(kwargs['idx']))

    monkeypatch.setattr(active, '_gen_active_config', _gen)
    _install_fake_context(monkeypatch)
    monkeypatch.setattr(Config, 'n_cores', 2)
    monkeypatch.setattr(Config, 'process_timeout', 30)

    mlp = test_potential()

    _run_add_active_configs(mlp, n_configs=2)

    assert mlp.n_train == 1
    assert any('bad frame' in message for message in mlp_caplog.messages)
    assert (
        '1/2 active learning workers failed or timed out'
        in mlp_caplog.messages
    )


@work_in_tmp_dir()
def test_add_active_configs_kills_timed_out_worker(
    mlp_caplog, monkeypatch, test_potential
):
    """A wedged worker is SIGTERMed, then SIGKILLed if it ignores SIGTERM.

    This is the regression that motivated the branch: previously the parent
    blocked forever in ``AsyncResult.get(timeout=None)``.
    """

    monkeypatch.setattr(
        active, '_gen_active_config', lambda *args, **kwargs: _config(0.0)
    )
    ctx = _install_fake_context(monkeypatch, hang=True, stubborn=True)
    monkeypatch.setattr(Config, 'n_cores', 1)
    monkeypatch.setattr(Config, 'process_timeout', 0.3)

    mlp = test_potential()

    _run_add_active_configs(mlp, n_configs=1)

    assert ctx.processes[0].calls == ['terminate', 'kill']
    assert mlp.n_train == 0
    assert any(
        'Timeout error when trying to generate an active configuration idx=0'
        in message
        for message in mlp_caplog.messages
    )
    assert (
        'All active learning workers failed or timed out; '
        'no new configurations generated this iteration' in mlp_caplog.messages
    )


@work_in_tmp_dir()
def test_add_active_configs_hard_caps_poll_loop(
    mlp_caplog, monkeypatch, test_potential
):
    """The poll loop force-kills workers even when process_timeout is None.

    Note the consequence: setting ``Config.process_timeout = None`` does not
    disable killing, it only raises the cap to the hard 7200 + 120 s default.
    """

    monkeypatch.setattr(
        active, '_gen_active_config', lambda *args, **kwargs: _config(0.0)
    )
    ctx = _install_fake_context(monkeypatch, hang=True)
    monkeypatch.setattr(active, 'time', _FakeClock(step=1000.0))
    monkeypatch.setattr(Config, 'n_cores', 1)
    monkeypatch.setattr(Config, 'process_timeout', None)

    mlp = test_potential()

    _run_add_active_configs(mlp, n_configs=1)

    assert ctx.processes[0].calls == ['kill']
    assert mlp.n_train == 0
    assert any(
        'Worker poll loop exceeded max duration' in message
        for message in mlp_caplog.messages
    )


@work_in_tmp_dir()
def test_add_active_configs_skips_missing_trajectory(
    mlp_caplog, monkeypatch, test_potential
):
    """keep_al_trajs must not raise when a timed-out worker wrote no .traj."""

    monkeypatch.setattr(
        active, '_gen_active_config', lambda *args, **kwargs: _config(0.0)
    )
    _install_fake_context(monkeypatch)
    monkeypatch.setattr(Config, 'n_cores', 1)
    monkeypatch.setattr(Config, 'process_timeout', 30)

    mlp = test_potential()

    _run_add_active_configs(mlp, n_configs=1, keep_al_trajs=True)

    assert (
        'Trajectory file not found for idx=0; skipping save'
        in mlp_caplog.messages
    )


# --------------------------------------------------------------------------
# HILLS validation and bias inheritance
# --------------------------------------------------------------------------


HILLS_HEADER = [
    '#! FIELDS time cv1 sigma_cv1 height biasf\n',
    '#! SET multivariate false\n',
    '#! SET kerneltype stretched-gaussian\n',
]


def _write_hills(fname, data_lines):
    """Write a minimal PLUMED-style HILLS file (height is column index 3)."""
    with open(fname, 'w') as hills_file:
        hills_file.writelines(HILLS_HEADER)
        hills_file.writelines(f'{line}\n' for line in data_lines)


@pytest.mark.parametrize(
    'fields, expected',
    [
        (['1.0', '0.5', '0.05', '0.2', '3'], True),
        (['1.0', '-0.5', '5e-2', '2E-1', '3'], True),
        (['1.0', '0.5', '0.05', 'nan', '3'], False),
        (['1.0', '0.5', '0.05', 'inf', '3'], False),
        (['1.0', '0.5', '0.05', '-inf', '3'], False),
        (['1.0', '0.5', '0.05', 'height', '3'], False),
        ([], True),
    ],
)
def test_is_valid_hills_line(fields, expected):
    assert active._is_valid_hills_line(fields) is expected


def test_is_valid_hills_line_checks_column_count():
    fields = ['1.0', '0.5', '0.05', '0.2', '3']

    assert active._is_valid_hills_line(fields, expected_ncols=5) is True
    assert active._is_valid_hills_line(fields, expected_ncols=4) is False


@work_in_tmp_dir()
def test_generate_inheritable_metad_bias_skips_missing_and_empty(
    mlp_caplog, monkeypatch
):
    """Missing or header-only HILLS files are dropped, the rest still used.

    Before this branch a single missing file silently disabled bias
    inheritance for the whole iteration.
    """

    calls = {}

    def _spy(n_configs, hills_files, iteration, bias_start_iter):
        calls['n_configs'] = n_configs
        calls['hills_files'] = hills_files

    monkeypatch.setattr(active, '_generate_inheritable_metad_bias_hills', _spy)

    # idx 0 has data, idx 1 is header-only, idx 2 was never written
    _write_hills('HILLS_1_0.dat', ['   1.0   0.5   0.05   0.2   3'])
    _write_hills('HILLS_1_1.dat', [])

    active._generate_inheritable_metad_bias(
        n_configs=3, kwargs={'iteration': 1, 'bias_start_iter': 0}
    )

    assert calls == {'n_configs': 1, 'hills_files': ['HILLS_1_0.dat']}
    assert any(
        'Missing HILLS files detected' in message
        and 'HILLS_1_2.dat' in message
        for message in mlp_caplog.messages
    )
    assert any(
        'Empty HILLS files detected' in message and 'HILLS_1_1.dat' in message
        for message in mlp_caplog.messages
    )


@work_in_tmp_dir()
def test_generate_inheritable_metad_bias_gives_up_when_all_empty(
    mlp_caplog, monkeypatch
):
    """With no usable HILLS data the delegate is not called at all."""

    def _should_not_run(*args, **kwargs):
        raise AssertionError('should not generate bias from empty HILLS')

    monkeypatch.setattr(
        active, '_generate_inheritable_metad_bias_hills', _should_not_run
    )

    _write_hills('HILLS_1_0.dat', [])

    active._generate_inheritable_metad_bias(
        n_configs=1, kwargs={'iteration': 1, 'bias_start_iter': 0}
    )

    assert (
        'No non-empty HILLS files were found for generating '
        'inheritable metadynamics bias' in mlp_caplog.messages
    )


@work_in_tmp_dir()
def test_generate_inheritable_metad_bias_hills_filters_bad_lines(mlp_caplog):
    """Truncated and NaN gaussians are dropped; heights are averaged."""

    _write_hills(
        'HILLS_1_0.dat',
        [
            '   1.0   0.5   0.05   0.2   3',
            '   2.0   0.6   0.05   0.2   3',
            # PLUMED sometimes fails to flush the final line
            '   3.0   0.7',
        ],
    )
    _write_hills(
        'HILLS_1_1.dat',
        [
            '   1.0   0.8   0.05   0.2   3',
            # A diverged trajectory poisons the height with NaN
            '   2.0   0.9   0.05   nan   3',
            '   3.0   1.0   0.05   0.2   3',
        ],
    )

    active._generate_inheritable_metad_bias_hills(
        n_configs=2,
        hills_files=['HILLS_1_0.dat', 'HILLS_1_1.dat'],
        iteration=1,
        bias_start_iter=1,
    )

    with open('HILLS_1.dat', 'r') as f:
        lines = [line for line in f if not line.startswith('#!')]

    # 2 valid from file 0 (truncated line popped) + 2 from file 1 (NaN skipped)
    assert len(lines) == 4

    heights = [float(line.split()[3]) for line in lines]
    assert heights == pytest.approx([0.1, 0.1, 0.1, 0.1])

    assert any(
        'Truncated last line detected in HILLS_1_0.dat' in message
        for message in mlp_caplog.messages
    )
    assert any(
        'Skipped 1 invalid/NaN line(s) in HILLS_1_1.dat' in message
        for message in mlp_caplog.messages
    )

    assert not os.path.exists('HILLS_1_0.dat')
    assert os.path.exists('accumulated_bias/bias_after_iter_1.dat')


# --------------------------------------------------------------------------
# plumed sum_hills failure handling
# --------------------------------------------------------------------------


def _stub_bias():
    """Minimal duck-typed PlumedBias for _generate_grid_from_hills."""
    cv = SimpleNamespace(name='cv1')

    return SimpleNamespace(
        cvs=[cv], metad_cvs=[cv], width=[0.05], n_metad_cvs=1
    )


def _stub_configurations():
    return SimpleNamespace(plumed_coordinates=np.array([[0.5], [1.5]]))


class _StubPopen:
    """Stand-in for subprocess.Popen that fakes a `plumed sum_hills` run."""

    def __init__(self, returncode, grid_content=None):
        self.returncode = returncode
        self.grid_content = grid_content

    def __call__(self, args):
        if self.grid_content is not None:
            outfile = args[args.index('--outfile') + 1]
            with open(outfile, 'w') as f:
                f.write(self.grid_content)

        return SimpleNamespace(wait=lambda: self.returncode)


@work_in_tmp_dir()
@pytest.mark.parametrize(
    'returncode, grid_content, expected',
    [
        # plumed crashed on corrupt HILLS data
        (1, None, False),
        # plumed exited cleanly but wrote nothing usable
        (0, None, False),
        (0, '', False),
        (0, '0.5 -1.0\n1.5 -2.0\n', True),
    ],
)
def test_generate_grid_from_hills_return_value(
    monkeypatch, returncode, grid_content, expected
):
    monkeypatch.setattr(active, 'Popen', _StubPopen(returncode, grid_content))

    result = active._generate_grid_from_hills(
        configurations=_stub_configurations(), iteration=1, bias=_stub_bias()
    )

    assert result is expected


@work_in_tmp_dir()
def test_attach_inherited_bias_falls_back_to_zero(mlp_caplog, monkeypatch):
    """A failed grid generation degrades to zero bias, it does not raise.

    Previously ``np.loadtxt`` was called unconditionally and blew up on the
    missing/empty grid file, taking the whole AL iteration with it.
    """

    monkeypatch.setattr(
        active, '_generate_grid_from_hills', lambda **kwargs: False
    )

    with open('HILLS_0.dat', 'w') as f:
        f.write('#! FIELDS time cv1 sigma_cv1 height biasf\n')

    configurations = ConfigurationSet()
    for x in (0.0, 1.0):
        configurations.append(_config(x, energy=-1.0 - x))

    active._attach_inherited_bias_energies(
        configurations=configurations,
        iteration=1,
        bias_start_iter=0,
        bias=_stub_bias(),
    )

    assert all(config.energy.inherited_bias == 0 for config in configurations)
    assert any(
        'Falling back to zero inherited bias' in message
        for message in mlp_caplog.messages
    )


# --------------------------------------------------------------------------
# Real multiprocessing
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_add_active_configs_spawn_integration(tmp_path):
    """Exercise _add_active_configs with real multiprocessing spawn.

    Unit tests can fake most behavior, but they cannot catch spawn-specific
    issues such as pickling the process target, queue communication across
    processes, or child-process imports. The driver lives in
    tests/integration_scripts/spawn_active.py so that ruff and ty analyse it
    like any other module; it patches the worker target to a cheap top-level
    function and lets the production _add_active_configs create real workers.
    """

    here = Path(__file__).resolve().parent
    script = here / 'integration_scripts' / 'spawn_active.py'
    assert script.exists(), script

    repo_root = here.parent
    env = os.environ.copy()

    # The subprocess runs from tmp_path so its generated datasets do not touch
    # the repository. Add the repo to PYTHONPATH so it imports the checkout
    # under test rather than requiring an installed wheel.
    env['PYTHONPATH'] = (
        f'{repo_root}{os.pathsep}{env["PYTHONPATH"]}'
        if env.get('PYTHONPATH')
        else str(repo_root)
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    # The subprocess prints a small JSON summary once the real parent process
    # has drained both child results and saved the per-iteration dataset.
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        'n_train': 2,
        'dataset_path': 'datasets/dataset_after_iter_0.npz',
    }
    assert (tmp_path / payload['dataset_path']).exists()
