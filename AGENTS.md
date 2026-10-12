# AGENTS.md

## Environment setup

Primary environment is managed by **pixi** (Linux-only — `platforms = ["linux-64"]`). On a machine without a GPU, set `CONDA_OVERRIDE_CUDA=12.0` during install or the CUDA-locked conda packages will fail to resolve:

```bash
CONDA_OVERRIDE_CUDA=12.0 pixi install -e mace
```

MACE also still has a conda install path (`./install_mace.sh` / `environment_mace.yml`), kept working alongside pixi during the transition. Both are tested in CI.

The ACE backend uses a separate conda environment (`environment_ace.yml`) and requires Julia ≥ 1.6 pre-installed in `$PATH` before running `./install_ace.sh`.

Enable pre-commit once after cloning:
```bash
pre-commit install
```

## Developer commands

```bash
# Run all tests (MACE env)
pixi run -e mace test
# Equivalent: pytest --cov=mlptrain --cov-report term --cov-report xml

# Run a specific file or test
pixi run -e mace pytest tests/test_mace.py
pixi run -e mace pytest tests/test_mace.py::test_train_logging
pixi run -e mace pytest -k "test_name"

# Lint + format + typecheck (all at once)
pixi run -e mace pre-commit run --all-files

# Individual tools (within mace env)
ruff check --fix mlptrain/ tests/ examples/
ruff format mlptrain/ tests/ examples/
ty check mlptrain/ tests/ examples/

# Build docs
pixi run -e docs
```

## Critical code conventions

**Banned top-level imports (Ruff TID253 — enforced by pre-commit):** `mace`, `openmm`, `openmmml`, `sklearn`, and `torch` must never be imported at module level. Always import them lazily inside function/method bodies:

```python
def ase_calculator(self):
    from mace.calculators import MACECalculator  # correct — lazy
    ...
```

Violations will cause pre-commit to exit non-zero even after auto-fixing; re-stage and re-commit the auto-fixed changes.

**Type checker is `ty`, not mypy.** Use `# ty:ignore[rule-name]` not `# type: ignore`. Strict mode applies only to files listed in `[tool.ty.overrides]` in `pyproject.toml`.

**Ruff style:** line length = 79, single-quoted strings. Not the more common 88/black defaults.

**Units:** distance Å, energy eV, force eV Å⁻¹, time fs. Not SI, kJ/mol, nm, or ps.

**`Config` is a module-level mutable singleton** (`mlptrain.Config`). Changes to `Config.mace_params`, `Config.orca_keywords`, etc. persist for the entire process. Tests must restore any mutations.

## Architecture

Single Python package `mlptrain`. Not a monorepo. No separate build step needed — editable install via pixi.

Key directories:
- `mlptrain/potentials/mace/` — MACE backend (primary, GPU-accelerated)
- `mlptrain/potentials/ace/` — ACE backend (legacy, Julia/pyjulip)
- `mlptrain/sampling/` — ASE-driven MD, OpenMM MD, umbrella sampling, metadynamics, PLUMED interface
- `mlptrain/training/active.py` — active learning loop (`al_train`)
- `mlptrain/configurations/calculate.py` — QM interface (autode wrapping ORCA, G16, G09, XTB)

Active learning runs a nested process tree: `_add_active_configs` starts one raw `mp.Process` per configuration (spawn context, results returned over an `mp.Queue`), each of which runs MD that may itself start PLUMED/ORCA subprocesses or a metadynamics `ProcessPoolExecutor`. Everything crossing a process boundary must be picklable under spawn, which constrains how backends and calculators are structured. Do not go back to `mp.Pool` — its workers are daemonic and cannot have children. Switching the start method does not help: `daemon = True` is set by `Pool` for every context, and `fork` additionally breaks the parent's live CUDA context and its threaded torch/OpenMP state. See `docs/dev/parallelism.rst`.

Three timeouts bound that tree: `Config.dynamics_timeout` (SIGALRM inside `run_with_timeout`, 2 h), `Config.process_timeout` (parent SIGTERM→SIGKILL per worker, 8 h), and a hard poll-loop cap of `process_timeout + 120 s`. Timeouts degrade rather than raise: `run_mlp_md`, `_gen_active_config` and `Metadynamics._run_single_metad` return `None`, and callers must check.

## Testing quirks

- `test_mace.py` silently skips all tests if `mace` is not importable — always run in the mace pixi env to get real coverage.
- OpenMM tests are named with `test_openmm` in the test name; ACE CI excludes them with `-k "not test_openmm"`.
- One umbrella test is permanently skipped on GHA (`@pytest.mark.skip(reason='Test fails on GHA with MACE')`).
- Many tests use `@work_in_tmp_dir()` or `@work_in_zipped_dir(path)` decorators (from `tests/data/utils.py`) that change CWD to a temp directory and clean up after. Tests that write files assume CWD is writable and temporary.
- `pytest addopts` always emits `coverage.xml` — every test run writes this file.
- The `test_potential` fixture yields a factory callable `_create_potential(name, calculator, system)`, not a direct instance.

## Environment variables

| Variable | Purpose |
|---|---|
| `CONDA_OVERRIDE_CUDA=12.0` | Required during `pixi install` on non-GPU machines |
| `MLT_LOG_LEVEL` | Runtime log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`); default `INFO` |

Tests marked `@pytest.mark.integration` spawn real subprocesses; deselect them with `-m "not integration"`.
