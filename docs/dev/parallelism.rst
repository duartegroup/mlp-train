**********************************
Parallelism and timeouts
**********************************

Active learning in ``mlp-train`` runs a *nested* tree of processes: the
active-learning loop starts one worker per requested configuration, each
worker drives its own molecular dynamics, and that dynamics may in turn
start further processes (PLUMED, a QM code, or a pool of metadynamics
walkers). This page describes that structure, the three timeouts that bound
it, and the constraints it places on contributors.

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

=====================================
Why not ``multiprocessing.Pool``
=====================================

Both ``_add_active_configs`` and ``Metadynamics`` previously used
``mp.get_context('spawn').Pool(...)``. Two properties of ``Pool`` made that
untenable:

1. **Pool workers are daemonic.** A daemonic process is not allowed to have
   children, so any nesting in the tree above is illegal from inside a pool
   worker. ``concurrent.futures.ProcessPoolExecutor`` creates non-daemonic
   workers, which is why ``Metadynamics`` now uses it.

2. **There is no way to reclaim a wedged worker.** The collection loop called
   ``AsyncResult.get(timeout=None)``, so a single non-terminating trajectory
   blocked the parent indefinitely — the whole active-learning run stalled
   with no log output and no way to recover short of killing the job.

``_add_active_configs`` therefore manages raw ``mp.Process`` objects and a
``mp.Queue`` directly, which gives the parent the ability to poll, time out,
and kill.

====================================
Why not the ``fork`` start method
====================================

Switching from ``spawn`` to ``fork`` is a recurring suggestion, usually on the
grounds that ``spawn``'s pickling requirement is inconvenient and that
mlp-train targets Linux HPC anyway. It does not help, and it breaks things.

**It does not lift the daemonic restriction.** Daemonic-ness is a property of
``Pool``, not of the start method: ``multiprocessing/pool.py`` sets
``w.daemon = True`` unconditionally, for every context.
``concurrent.futures.process`` never sets it at all. A forked ``Pool`` worker
is just as unable to have children as a spawned one — the move to
``mp.Process`` and ``ProcessPoolExecutor`` was the actual fix, and it is
required regardless of start method.

**It breaks CUDA.** ``al_train`` calls ``mlp.train()`` in the parent before
entering the iteration loop, so by the time workers are created the parent
holds a live CUDA context (the MACE backend calls
``torch.cuda.empty_cache()``). A forked child inherits that context in an
unusable state and raises ``Cannot re-initialize CUDA in forked subprocess``.
This is a Linux problem, not a macOS one.

**It breaks threaded libraries.** ``fork`` copies only the calling thread but
all of the process's locks, in whatever state they happened to be in. A parent
running MACE under OpenMP/MKL threadpools therefore produces children that
deadlock intermittently and unreproducibly. CPython 3.12 emits a
``DeprecationWarning`` for forking a multi-threaded process, and 3.14 changes
the Linux default start method to ``forkserver``.

(``fork`` combined with matplotlib also fails on macOS, but that is the least
of the reasons.)

=====================
The three timeouts
=====================

.. list-table::
   :header-rows: 1
   :widths: 14 30 24 12 40

   * - Layer
     - Mechanism
     - Knob
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

The inner timeout is **best effort**. ``SIGALRM`` is delivered to the main
thread of a process, but it can be masked or swallowed entirely by
C-extension code — PLUMED and PyTorch both do this in places. When that
happens the alarm never fires and the trajectory runs to completion. The
per-worker hard kill exists precisely because the inner timeout cannot be
relied on alone; do not remove one on the grounds that the other covers it.

Why both layers are needed
--------------------------

The inner timeout is not a faster version of the outer one, and the two are
not interchangeable. Two reasons.

**Only active learning has a parent watching.** ``run_mlp_md`` is called from
five places, and the poll loop covers exactly one of them:

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

Deleting the inner timeout would re-open the original hang for metadynamics,
umbrella sampling, width estimation and τ_acc. Those four sites have no
parent-side kill; closing that gap is unfinished work.

**A timeout returns, a kill does not.** When the inner timeout fires,
``run_mlp_md`` unwinds normally: ``work_in_tmp_dir``'s ``finally`` block still
copies ``kept_substrings`` (``.traj``, ``.dat``) back out of the temporary
directory, and ``PlumedCalculator.finalize()`` still runs. A ``SIGKILL`` from
the parent loses both — the temp directory is orphaned, ``keep_al_trajs``
produces nothing for that worker, and the ``HILLS`` file it was writing never
arrives for bias inheritance. The per-worker kill is the backstop for when the
graceful path fails, not the preferred path.

.. warning::

   Setting ``Config.process_timeout = None`` does **not** disable killing. It
   disables the per-worker check, but the poll loop still force-kills every
   remaining worker once its hard cap is reached, which falls back to
   ``7200 + 120`` seconds when no timeout is configured.

Both knobs are plain attributes on the ``Config`` singleton::

    import mlptrain as mlt

    mlt.Config.dynamics_timeout = 15 * 60   # 15 minutes per trajectory
    mlt.Config.process_timeout = 60 * 60    # 1 hour per worker

======================
The result protocol
======================

Workers never return values; they put a four-tuple on a shared
``mp.Queue``::

    (idx, 'ok', configuration_or_None, None)
    (idx, 'error', None, repr(exception))

``_gen_active_config_worker`` catches ``BaseException`` so that a failure
inside one worker is reported to the parent rather than lost across the
process boundary.

The parent drains the queue **inside** the poll loop, on every iteration —
not only after all workers have exited. This is not an optimisation. A
``mp.Queue`` is backed by an OS pipe with a finite buffer; a child that has
written more than the buffer holds blocks in its feeder thread until the
parent reads, and a child blocked that way never exits. Joining before
draining is the classic multiprocessing queue/join deadlock. If you add
anything to the poll loop, keep the drain unconditional.

===============================
Constraints for contributors
===============================

**Everything crossing a process boundary must be picklable under spawn.**
There is no shared memory and no inherited state: the child re-imports
``mlptrain`` from scratch. This is why the worker arguments are built with
``init_config.copy()``, ``mlp.copy()`` and ``selection_method.copy()``, and
why backends must keep their calculators constructible from picklable state.

**Config does not propagate.** ``Config`` is a module-level mutable singleton,
so each spawned child gets a freshly imported copy with the defaults.
Mutating ``Config`` in the parent after workers have started has no effect on
them; anything a worker needs must be passed through its arguments.

**Failures degrade, they do not raise.** A worker that times out, crashes, or
whose trajectory is abandoned contributes ``None``. The iteration continues
with fewer configurations and logs how many were lost. Functions along this
path return ``Optional`` for that reason — ``run_mlp_md``,
``_run_mlp_md``, ``Metadynamics._run_single_metad`` and
``_gen_active_config`` all may return ``None``, and callers must check.

**Corrupt PLUMED output is expected, not exceptional.** A trajectory that
diverges before being killed can leave a ``HILLS`` file that is empty,
truncated mid-line, or contains ``NaN`` gaussians. Bias inheritance skips
missing and empty files, drops invalid lines, and falls back to zero
inherited bias if ``plumed sum_hills`` fails, rather than aborting the
iteration.

==================
Tuning guidance
==================

``_add_active_configs`` splits the available cores across workers::

    n_processes = min(n_configs, Config.n_cores)
    n_cores_pp  = max(Config.n_cores // n_configs, 1)

``Config.n_cores`` must be a multiple of ``n_configs_iter`` when it exceeds
it, otherwise ``_add_active_configs`` raises. Choosing
``Config.n_cores == n_configs_iter`` gives one core per worker and the
simplest behaviour.

The default ``dynamics_timeout`` of 2 hours is very generous for active
learning, as each step takes ``2 + n_calls**3 + extra_time`` fs, which is typically
tens to hundreds of femtoseconds. If your trajectories normally finish in
seconds, it is safe to set '' dynamics_timeout `` to a few minutes instead. The ``process_timeout``
should be kept above ``dynamics_timeout``: the worker also has to run selection
and, for the last frame, a single-point QM calculation. For large or difficult-to-converge systems, you might even consider increasing this above the default to avoid killing your jobs mid-convergence.  
