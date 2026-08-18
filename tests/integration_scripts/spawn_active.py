"""Drive ``_add_active_configs`` through real ``spawn`` multiprocessing.

Run as a subprocess by ``tests/test_active.py``, not by pytest. The unit
tests fake the process context, so they cannot catch spawn-specific problems:
pickling the process target, queue communication between processes, or the
child re-importing the module. This script exercises all three against the
real implementation, replacing only the expensive per-worker payload.

Prints a one-line JSON summary on success. Requires the repository on
``PYTHONPATH`` and should be run from a scratch directory, since
``_add_active_configs`` writes ``datasets/`` relative to the CWD.
"""

import json
import multiprocessing as mp

from autode.atoms import Atom

from mlptrain.config import Config
from mlptrain.configurations import Configuration
from mlptrain.potentials import MLPotential
from mlptrain.training import active
from mlptrain.training.selection import AbsDiffE, SelectionMethod


def make_config(idx: int) -> Configuration:
    config = Configuration(atoms=[Atom('H', float(idx), 0.0, 0.0)])
    config.energy.true = -float(idx + 1)

    return config


def worker(
    result_queue: 'mp.queues.Queue',
    idx: int,
    config: Configuration,
    mlp: MLPotential,
    selector: SelectionMethod,
    n_cores: int,
    kwargs: dict,
) -> None:
    """Cheap stand-in for ``_gen_active_config_worker``.

    The signature must match the real worker exactly — it is assigned over it
    below, and the type checker enforces that the replacement is compatible.
    Mirrors the real protocol: put ``(idx, status, config, error)`` on the
    multiprocessing queue for the parent to drain.
    """
    result_queue.put((idx, 'ok', make_config(idx), None))


class DummyPotential(MLPotential):
    """``_add_active_configs`` only needs a copyable potential with
    ``training_data``; the calculator and training hooks are never reached."""

    @property
    def ase_calculator(self):
        raise NotImplementedError

    @property
    def requires_atomic_energies(self) -> bool:
        return False

    @property
    def requires_non_zero_box_size(self) -> bool:
        return False

    def _train(self) -> None:
        return None


def main() -> None:
    Config.n_cores = 2
    Config.process_timeout = 10

    # _add_active_configs reads the worker target at process-creation time,
    # so the spawned children run this function instead of the real one.
    # ty flags any rebinding of a module-level function as implicit shadowing;
    # the signature above is kept identical so the substitution stays honest.
    active._gen_active_config_worker = worker  # ty:ignore[invalid-assignment]

    mlp = DummyPotential(
        'spawn_active',
        system=None,  # ty: ignore[invalid-argument-type]
    )
    active._add_active_configs(
        mlp,
        init_config=make_config(99),
        selection_method=AbsDiffE(),
        n_configs=2,
        method_name='mock',
        iteration=0,
        inherit_metad_bias=False,
        bias_start_iter=0,
        max_time=1.0,
    )

    print(
        json.dumps(
            {
                'n_train': mlp.n_train,
                'dataset_path': 'datasets/dataset_after_iter_0.npz',
            }
        )
    )


# Required: under spawn the children re-import this file as __mp_main__, and
# without the guard each would re-run main() and fork-bomb the test.
if __name__ == '__main__':
    main()
