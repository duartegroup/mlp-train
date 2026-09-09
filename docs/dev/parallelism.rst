*******************************************
Parallelism and timeouts in active learning
*******************************************

Active learning in ``mlp-train`` runs a *nested* tree of processes: the
active-learning loop starts one worker per requested configuration, each
worker drives its own molecular dynamics, and that dynamics may in turn
start further processes (PLUMED, a QM code, or a pool of metadynamics
walkers). This page describes that structure, the three timeouts that bound
it, and what to keep in mind when writing code that runs inside it.

=================
The process tree
=================

.. code-block:: text

    al_train()                                     parent process
      └─ _add_active_configs()
           ├─ mp.Process  worker idx=0             one per configuration
           │    └─ _gen_active_config()
           │         └─ run_mlp_md() / run_mlp_md_openmm()
           │              ├─ SIGALRM timer         inner timeout
           │              └─ Popen(...)            PLUMED driver, ORCA (autodE)
           ├─ mp.Process  worker idx=1
           ├─ ...
           └─ mp.Queue                             (idx, status, config, error)

``Metadynamics`` adds another level when it is driven from within a worker:

.. code-block:: text

    Metadynamics.run_metadynamics()
      └─ ProcessPoolExecutor(mp_context=spawn)
           └─ _run_single_metad()  ×  n_runs
                └─ run_mlp_md()  →  PLUMED

All process creation uses the ``spawn`` start method.

=======================================
Why ``mp.Process`` and the executor
=======================================

Two properties of the tree above decide how workers are launched.

**Workers must be able to have children.** A metadynamics worker starts
PLUMED, and an active-learning worker may start a QM code, so no level of the
tree can be a leaf. ``concurrent.futures.ProcessPoolExecutor`` creates
non-daemonic workers, which are free to do this, and it is what
``Metadynamics`` uses. `

**The parent must be able to reclaim an individual worker.**
``_add_active_configs`` manages raw ``mp.Process`` objects and an ``mp.Queue``
directly, so it can poll each worker, notice one that has outrun its timeout,
and terminate just that one while the rest of the iteration carries on.

=================================
Why the ``spawn`` start method
=================================

Every child is started with ``spawn``: a fresh interpreter that imports
``mlptrain`` from scratch and inherits nothing from the parent's memory. That
matters because by the time workers are created the parent is holding state
that does not survive being copied.

**A live CUDA context.** ``al_train`` calls ``mlp.train()`` before entering
the iteration loop, and the MACE backend calls ``torch.cuda.empty_cache()``,
so the parent holds an initialised CUDA context. ``spawn`` gives each child a
clean interpreter that initialises CUDA for itself, which is the only way for
a child to use the GPU at all.

**Unification across platforms.** ``spawn`` is the only start method available on
all supported platforms, so contributors developing on macOS exercise the same
process semantics that CI and Linux HPC do.

The cost is that everything crossing a process boundary has to be picklable,
and that ``Config`` is rebuilt from the import in each child. Both are made
concrete in `Important caveats for code that runs in a worker`_.

=====================
The three timeouts
=====================

.. list-table::
   :header-rows: 1
   :widths: 14 30 24 12 40

   * - Layer
     - Mechanism
     - Adjuster
     - Default
     - Catches
   * - Inner
     - ``SIGALRM`` via ``run_with_timeout``
     - ``Config.dynamics_timeout``
     - 2 h
     - A single ``dyn.run()`` that will not converge
   * - Per-worker
     - Parent poll loop: ``terminate()`` then ``kill()``
     - ``Config.process_timeout``
     - 8 h
     - A worker wedged anywhere, including where ``SIGALRM`` never fires
   * - Loop
     - Hard cap of ``process_timeout + 120 s``
     - *(none)*
     - —
     - The poll loop itself failing to make progress

The inner timeout is the graceful layer. ``SIGALRM`` is delivered to the main
thread, ``run_with_timeout`` raises, and ``run_mlp_md`` unwinds through its
normal ``finally`` blocks. C-extension code — PLUMED and PyTorch both do this
in places — can hold a signal until it returns to the interpreter, so a
trajectory deep inside such a call may run past ``dynamics_timeout``. When it
does, the per-worker timeout is the layer that still delivers the bound: the
parent terminates the worker and the iteration moves on.

What each layer covers
----------------------

The two layers bound different things, and the inner one covers considerably
more of the codebase.

**The parent poll loop is specific to active learning.** ``run_mlp_md`` is
called from five places, and the poll loop supervises one of them:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Call site
     - Bounded by
   * - ``_gen_active_config`` (``training/active.py``)
     - Inner timeout **and** the parent poll loop
   * - ``Metadynamics._run_single_metad``
     - Inner timeout only — ``future.result()`` takes no timeout
   * - ``Metadynamics._get_width_for_single``
     - Inner timeout only — ``AsyncResult.get()`` takes no timeout
   * - ``UmbrellaSampling._run_individual_window``
     - Inner timeout only — ``AsyncResult.get()`` takes no timeout
   * - ``TauCalculator._calculate_single``
     - Inner timeout only — no worker process at all

Metadynamics, umbrella sampling, width estimation and τ_acc are therefore
bounded by ``Config.dynamics_timeout`` alone: they run with no supervising
parent, so the inner timeout is what keeps them finite.

**A timeout returns, a kill does not.** When the inner timeout fires,
``run_mlp_md`` unwinds normally: ``work_in_tmp_dir``'s ``finally`` block still
copies ``kept_substrings`` (``.traj``, ``.dat``) back out of the temporary
directory, and ``PlumedCalculator.finalize()`` still runs, so the partial
trajectory and the ``HILLS`` file written so far survive. A ``SIGKILL`` from
the parent trades those for a guaranteed bound — the temporary directory is
orphaned, ``keep_al_trajs`` produces nothing for that worker, and no ``HILLS``
file arrives for bias inheritance.

.. note::

   The poll loop always terminates. Setting ``Config.process_timeout = None``
   turns off the per-worker check, and the loop's own cap —
   ``process_timeout + 120 s``, or ``7200 + 120`` seconds when no per-worker
   timeout is configured — still bounds the iteration and kills whatever is
   left running.

Both knobs are plain attributes on the ``Config`` singleton, set at module
scope so that spawned workers pick them up (see
`Important caveats for code that runs in a worker`_)::

    import mlptrain as mlt

    mlt.Config.dynamics_timeout = 15 * 60   # 15 minutes per trajectory
    mlt.Config.process_timeout = 60 * 60    # 1 hour per worker

======================
The result protocol
======================

A spawned worker has no way to hand a return value back to its parent, so
``_gen_active_config_worker`` puts a four-tuple on a shared ``mp.Queue``
instead::

    (idx, 'ok', configuration_or_None, None)
    (idx, 'error', None, repr(exception))

``idx`` says which worker the result came from, so the parent can match it to
the process it started. The worker catches ``BaseException`` before building
the tuple, so a failure inside one worker reaches the parent as an ``'error'``
entry rather than disappearing along with the process.

.. admonition:: Why does the parent read the queue on every pass?

  The obvious way to collect results is "wait for the workers to finish, then
  read what they left on the queue". That order deadlocks, and why it does is
  the reason the poll loop is shaped the way it is.

  An ``mp.Queue`` is not a shared list. Behind it sits an OS pipe with a
  fixed-size buffer, and a background *feeder thread* inside the worker that
  copies queued items into that pipe:

  .. code-block:: text

      worker                                        parent

      queue.put(result) → feeder thread → [ pipe buffer ] → queue.get()
                                            fixed size

  If nothing is reading the parent's end, that buffer fills up. The feeder
  thread then blocks part-way through its write, and a process cannot exit while
  its feeder thread is still trying to flush — so the worker hangs, having
  already done all of its work, one step short of exiting. A parent sitting in
  ``worker.join()`` waiting for that exit waits forever, because each side is
  now waiting on the other. This is the classic multiprocessing queue/join
  deadlock.

  ``_add_active_configs`` sidesteps it by draining the queue **inside** the poll
  loop, on every pass, instead of after the workers have been joined. Reading
  keeps the pipe emptying, the feeder thread never blocks, and each worker exits
  as soon as its result is safely on the queue.

  If you add anything to the poll loop, keep the drain unconditional — it also
  has to run on the passes where no worker has finished.

================================================
Important caveats for code that runs in a worker
================================================

**Everything crossing a process boundary must be picklable under spawn.**
There is no shared memory and no inherited state: the child re-imports
``mlptrain`` from scratch. This is why the worker arguments are built with
``init_config.copy()``, ``mlp.copy()`` and ``selection_method.copy()``, and
why backends must keep their calculators constructible from picklable state.

**Set Config attributes at module scope.** ``Config`` is a module-level
singleton, and each spawned worker builds its own by re-importing — including
re-importing the script it was launched from. That re-import runs everything
at module scope but skips the ``if __name__ == '__main__':`` block, which is
exactly the split you want::

    import mlptrain as mlt

    # Module scope: re-executed on import inside every spawned worker,
    # so these are the values the workers actually run with.
    mlt.Config.n_cores = 10
    mlt.Config.orca_keywords = ['PBE', 'def2-SVP', 'EnGrad']
    mlt.Config.dynamics_timeout = 15 * 60

    if __name__ == '__main__':
        # Parent only: not re-run by the workers.
        system = mlt.System(mlt.Molecule('methane.xyz'), box=None)
        mlp = mlt.potentials.MACE('methane', system=system)
        mlp.al_train(method_name='orca', temp=1000)

Moving ``mlt.Config.dynamics_timeout = 15 * 60`` inside the ``__main__``
guard would silently leave every worker on the 2 hour default. The same goes
for assignments made after ``al_train`` has started its workers: those reach
the parent only. Anything else a worker needs is passed explicitly through its
arguments.

``examples/methane.py`` uses this layout — ``Config`` at the top of the file,
everything else under the guard.

**A lost worker costs configurations, not the iteration.** A worker whose
trajectory hit ``dynamics_timeout``, that was terminated at
``process_timeout``, or that raised and reported through the
``(idx, 'error', …)`` tuple, contributes ``None`` to the pool of results. The
iteration continues with the configurations that did arrive and logs how many
trajectories were lost. Functions along this path return ``Optional`` for that
reason — ``run_mlp_md``, ``_run_mlp_md``, ``Metadynamics._run_single_metad``
and ``_gen_active_config`` all may return ``None``, and callers must check.

.. note::
  As of now the above is only true for active learning. Metadynamics, umbrella 
  sampling, and τ_acc are not supervised by a parent, so a worker that fails or 
  is killed will terminate the iteration. Therefore, if you are running 
  metadynamics, umbrella sampling, or any other use of metadynamics without 
  active learning, the default ``Config.dynamics_timeout`` is set to 100000 
  hours, which is effectively infinite for most use cases and therefore should 
  not interrupt your calculations.

**Bias inheritance tolerates incomplete PLUMED output.** A trajectory that
diverges before being killed can leave a ``HILLS`` file that is empty,
truncated mid-line, or contains ``NaN`` gaussians. Missing and empty files are
skipped and not included, and the average is computed without them. Files with
invalid lines are still used, but the invalid lines are discarded. If
``plumed sum_hills`` fails, the inherited bias drops to zero, rather than
aborting the iteration.

==================
Tuning guidance
==================

``_add_active_configs`` splits the available cores across workers::

    n_processes = min(n_configs, Config.n_cores)
    n_cores_pp  = max(Config.n_cores // n_configs, 1)

So that every worker is given the same number of cores, ``Config.n_cores``
must be an exact multiple of ``n_configs_iter`` whenever it is larger than it.
A value that is larger but not a multiple raises::

    NotImplementedError: Active learning is only implemented using an
    multiple of the number n_configs_iter. Please use n*<n_configs> cores.

Setting ``Config.n_cores == n_configs_iter`` gives one core per worker and is
the simplest choice.

The default ``dynamics_timeout`` of 100,000 hours is effectively infinite for 
active learning, but this is done temporarily so as to not affect default 
behaviour for other modes of operation (metadynamics etc.). A suitable 
dynamics_timeout can be set based on the expected runtime of each step, as each 
step takes ``2 + n_calls**3 + extra_time`` fs, which is typically tens to 
hundreds of femtoseconds. A reasonable suggestion for active learning would be 
to set ``dynamics_timeout`` to around 2 hours.

If your trajectories normally finish in seconds, it is also safe to set 
``dynamics_timeout`` to a few minutes instead. The ``process_timeout`` should 
be kept above ``dynamics_timeout``: the worker also has to run selection and, 
for the last frame, a single-point QM calculation. For large or 
difficult-to-converge systems, you might consider increasing this above the 
suggested 2 hour value to avoid killing your jobs mid-convergence.
