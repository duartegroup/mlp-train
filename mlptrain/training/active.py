from __future__ import annotations

import mlptrain
import os
import shutil
import numpy as np
import multiprocessing as mp
import time
from queue import Empty
from copy import deepcopy
from typing import Optional, Union, List
from subprocess import Popen
from ase import units as ase_units
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory as ASETrajectory
from mlptrain.config import Config
from mlptrain.sampling import PlumedBias
from mlptrain.sampling.md_openmm import run_mlp_md_openmm
from mlptrain.sampling.md import run_mlp_md
from mlptrain.training.selection import SelectionMethod, AbsDiffE
from mlptrain.configurations import ConfigurationSet
from mlptrain.log import logger
from mlptrain.box import Box


def _gen_active_config_worker(
    result_queue: 'mp.queues.Queue',
    idx: int,
    config: 'mlptrain.Configuration',
    mlp: 'mlptrain.potentials._base.MLPotential',
    selector: 'mlptrain.training.selection.SelectionMethod',
    n_cores: int,
    kwargs: dict,
) -> None:
    """Run one active-learning task and send result to the parent process."""
    # Deduplicate log handlers inherited from the parent process.
    # Each fork inherits all handlers; without this guard, every
    # successive iteration adds an extra copy → duplicate log lines.
    import logging

    root = logging.getLogger()
    seen = set()
    for h in list(root.handlers):
        key = (
            type(h),
            getattr(h, 'baseFilename', None),
            getattr(h, 'stream', None),
        )
        if key in seen:
            root.removeHandler(h)
        else:
            seen.add(key)

    pid = os.getpid()
    logger.info(f'Worker idx={idx} pid={pid} starting _gen_active_config')
    try:
        result = _gen_active_config(
            config,
            mlp,
            selector,
            n_cores,
            **kwargs,
        )
        logger.info(
            f'Worker idx={idx} pid={pid} _gen_active_config returned '
            f'(result is {"None" if result is None else "a Configuration"})'
        )
        result_queue.put((idx, 'ok', result, None))
        logger.info(f'Worker idx={idx} pid={pid} put result on queue')
    except BaseException as err:  # noqa: BLE001
        logger.error(f'Worker idx={idx} pid={pid} caught exception: {err!r}')
        result_queue.put((idx, 'error', None, repr(err)))
        logger.info(f'Worker idx={idx} pid={pid} put error on queue')


def train(
    mlp: 'mlptrain.potentials._base.MLPotential',
    method_name: str,
    selection_method: SelectionMethod = AbsDiffE(),
    max_active_time: float = 1000,
    n_configs_iter: int = 10,
    temp: float = 300.0,
    max_e_threshold: Optional[float] = None,
    max_active_iters: int = 50,
    n_init_configs: int = 10,
    init_configs: Optional['mlptrain.ConfigurationSet'] = None,
    fix_init_config: bool = False,
    bbond_energy: Optional[dict] = None,
    fbond_energy: Optional[dict] = None,
    init_active_temp: Optional[float] = None,
    min_active_iters: int = 1,
    bias_start_iter: int = 0,
    restart_iter: Optional[int] = None,
    inherit_metad_bias: bool = False,
    constraints: Optional[List] = None,
    bias: mlptrain.Bias | mlptrain.PlumedBias | None = None,
    md_program: str = 'ASE',
    pbc: bool = False,
    box_size: Optional[list] = None,
    keep_al_trajs: bool = False,
) -> None:
    """
    Train a system using active learning, by propagating dynamics using ML
    driven molecular dynamics (MD) and adding configurations based on some
    selection criteria. Loop looks something like

    Generate configurations -> train a MLP -> run MLP-MD -> frames
                                   ^                             |
                                   |________ calc true  _________


    Active learning will loop until either:
        (1) the iteration > max_active_iters
        (2) no configurations are found to add

    --------------------------------------------------------------------------
    Arguments:
        mlp: Machine learned potential to train

        method_name: Name of a reference method to use as the ground truth e.g.
                     dftb, orca, gpaw

        selection_method: Method used to select active learnt configurations

        max_active_time: (float) Maximum propagation time in the active
                            learning loop. Default = 1 ps

        n_configs_iter: (int) Number of configurations to generate per
                        active learning cycle

        temp: (float) Temperature in K to propagate active learning at -
              higher is better for stability but requires more training

        max_e_threshold: (float) Maximum relative energy threshold for
                           configurations to be added to the training data

        max_active_iters: (int) Maximum number of active learning
                          iterations to perform. Will break if we hit the
                          early stopping criteria

        n_init_configs: (int) Number of initial configurations to generate,
                        will be ignored if init_configs is not None

        init_configs: (gt.ConfigurationSet) A set of configurations from
                      which to start the active learning from

        fix_init_config: (bool) Always start from the same initial
                         configuration for the active learning loop.
                         If False then:
                            1. The structure with the lowest energy (true
                         energy) is used
                            2. If constraints and/or biases are attached, then
                         the structure with the lowest biased energy (true
                         energy + bias energy) is used
                            3. If using the option to inherit metadynamics
                         bias, then the structure with the lowest inherited
                         biased energy (true energy + inherited bias energy)
                         is used.

        bbond_energy: (dict | None) Additional energy to add to a breaking
                      bond. e.g. bbond_energy={(0, 1), 0.1} Adds 0.1 eV
                      to the 'bond' between atoms 0 and 1 as velocities
                      shared between the atoms in the breaking bond direction

        fbond_energy: (dict | None) As bbond_energy but in the direction to
                      form a bond

        init_active_temp: (float | None) Initial temperature for velocities
                          in the 'active' MD search for configurations

        min_active_iters: (int) Minimum number of active iterations to
                             perform

        bias_start_iter: (int) Iteration index at which the bias starts to be
                         applied. If the bias is PlumedBias, then UPPER_WALLS
                         and LOWER_WALLS are still applied from iteration 0

        restart_iter: (int | None) Iteration index at which to restart active
                                   learning

        inherit_metad_bias: (bool) If True metadynamics bias is inherited from
                            a previous iteration to the next during active
                            learning

        constraints: (List) List of ASE contraints to use in the dynamics
                            during active learning

        bias: (mlptrain.Bias | mlptrain.PlumedBias) Bias to add during MD
              simulations, useful for exploring under-explored regions in the
              dynamics

        md_program: (str) 'ASE' or 'OpenMM'

        pbc: (bool) If True, MLP-MD propagates with periodic boundary conditions.
              The solvent should be therefore placed in a box and not a sphere.
              The training data are still treated as clusters in
              electronic structure computations.

        box_size: (List | None) Size of the box where MLP-MD propagated.

        keep_al_trajs: (bool) If True, MLP-MD trajectories generated during AL phase are saved into new folder.
    """
    if md_program.lower() == 'openmm':
        if not isinstance(mlp, mlptrain.potentials.MACE):
            raise ValueError(
                'The OpenMM backend only supports the use of the MACE potential.'
            )

        if any([bias, fbond_energy, bbond_energy, constraints]):
            raise NotImplementedError(
                "The OpenMM backend does not support the use of the 'bias', "
                "'fbond_energy', 'bbond_energy', or 'constraints' arguments."
            )

    _check_bias(bias=bias, temp=temp, inherit_metad_bias=inherit_metad_bias)

    if keep_al_trajs is True:
        os.makedirs('al_trajectories', exist_ok=True)

    if pbc and box_size is None:
        raise ValueError('For PBC in MD, the box_size cannot be None')

    if restart_iter is not None:
        _initialise_restart(
            mlp=mlp,
            restart_iter=restart_iter,
            inherit_metad_bias=inherit_metad_bias,
        )
        init_config = mlp.training_data[0]

    elif init_configs is None:
        init_config = mlp.system.configuration
        _gen_and_set_init_training_configs(
            mlp=mlp, method_name=method_name, num=n_init_configs
        )

    else:
        init_config = init_configs[0]
        _set_init_training_configs(
            mlp=mlp, init_configs=init_configs, method_name=method_name
        )

    if isinstance(bias, PlumedBias) and not bias.from_file:
        _attach_plumed_coords_to_init_configs(
            init_configs=mlp.training_data, bias=bias
        )

    if mlp.requires_atomic_energies:
        mlp.set_atomic_energies(method_name=method_name)

    mlp.train()

    # Run the active learning loop, running iterative MLP-MD
    for iteration in range(max_active_iters):
        if restart_iter is not None and iteration <= restart_iter:
            continue
        if isinstance(bias, PlumedBias) and iteration >= bias_start_iter:
            extra_time = 0
        else:
            extra_time = mlp.training_data.t_min(-n_configs_iter)

        previous_n_train = mlp.n_train

        init_config_iter = _update_init_config(
            init_config=init_config,
            mlp=mlp,
            fix_init_config=fix_init_config,
            bias=bias,
            inherit_metad_bias=inherit_metad_bias,
            bias_start_iter=bias_start_iter,
            iteration=iteration,
        )

        _add_active_configs(
            mlp=mlp,
            init_config=init_config_iter,
            selection_method=selection_method,
            n_configs=n_configs_iter,
            method_name=method_name,
            temp=temp,
            max_time=max_active_time,
            bbond_energy=bbond_energy,
            fbond_energy=fbond_energy,
            init_temp=init_active_temp,
            extra_time=extra_time,
            constraints=constraints,
            bias=deepcopy(bias),
            inherit_metad_bias=inherit_metad_bias,
            bias_start_iter=bias_start_iter,
            iteration=iteration,
            md_program=md_program,
            pbc=pbc,
            box_size=box_size,
            keep_al_trajs=keep_al_trajs,
        )

        # Active learning finds no configurations
        if mlp.n_train == previous_n_train:
            if iteration >= min_active_iters:
                logger.info('No AL configurations found')
                break

            else:
                logger.info('No AL configurations found. Skipping training')
                continue

        # If required, remove high-lying energy configurations from the data
        if max_e_threshold is not None:
            mlp.training_data.remove_above_e(max_e_threshold)

        if mlp.training_data.has_a_none_energy:
            mlp.training_data.remove_none_energy()

        mlp.train()

    if inherit_metad_bias:
        _remove_last_inherited_metad_bias_file(max_active_iters)

    logger.info(f'Final dataset size = {mlp.n_train} Active learning = DONE')
    return None


def _add_active_configs(
    mlp: 'mlptrain.potentials._base.MLPotential',
    init_config: 'mlptrain.Configuration',
    selection_method: 'mlptrain.training.selection.SelectionMethod',
    n_configs: int = 10,
    **kwargs,
) -> None:
    """
    Add a number (n_configs) of configurations to the current training data
    based on active learning selection of MLP-MD generated configurations
    """
    if Config.n_cores > n_configs and Config.n_cores % n_configs != 0:
        raise NotImplementedError(
            'Active learning is only implemented using '
            'an multiple of the number n_configs_iter. '
            f'Please use n*{n_configs} cores.'
        )

    n_processes = min(n_configs, Config.n_cores)
    n_cores_pp = max(Config.n_cores // n_configs, 1)
    logger.info(
        'Searching for "active" configurations with '
        f'{n_processes} processes using {n_cores_pp} cores / process'
    )

    if 'bias' in kwargs and kwargs['iteration'] < kwargs['bias_start_iter']:
        logger.info(
            f'Iteration {kwargs["iteration"]}: the bias potential '
            'is not applied'
        )
        kwargs['bias'] = _remove_bias_potential(kwargs['bias'])

    configs = ConfigurationSet()
    worker_results = {}
    timeout_s = Config.process_timeout
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    workers = []
    start_times = {}

    for idx in range(n_configs):
        kwargs['idx'] = idx
        kwargs_single = deepcopy(kwargs)
        worker = ctx.Process(
            target=_gen_active_config_worker,
            args=(
                result_queue,
                idx,
                init_config.copy(),
                mlp.copy(),
                selection_method.copy(),
                n_cores_pp,
                kwargs_single,
            ),
        )
        worker.start()
        logger.info(
            f'Started active-learning worker idx={idx} pid={worker.pid}'
        )
        workers.append((idx, worker))
        start_times[idx] = time.monotonic()

    pending = {idx for idx, _ in workers}
    loop_start = time.monotonic()
    last_status_log = loop_start
    max_loop_s = (timeout_s or 7200) + 120  # safety: timeout + 2 min buffer
    logger.info(
        f'Entering worker poll loop. pending={sorted(pending)}, '
        f'process_timeout={timeout_s}s, max_loop={max_loop_s}s'
    )

    while pending:
        now = time.monotonic()

        # Periodic status logging (every 60s)
        if now - last_status_log > 60:
            elapsed_loop = now - loop_start
            alive_info = {i: w.is_alive() for i, w in workers if i in pending}
            logger.info(
                f'Worker poll loop: {elapsed_loop:.0f}s elapsed. '
                f'pending={sorted(pending)}, alive={alive_info}'
            )
            last_status_log = now

        # Safety: break out if the loop itself exceeds a hard limit
        if (now - loop_start) > max_loop_s:
            logger.error(
                f'Worker poll loop exceeded max duration '
                f'({max_loop_s}s). Force-killing remaining '
                f'workers: {sorted(pending)}'
            )
            for i, w in workers:
                if i in pending and w.is_alive():
                    logger.error(f'Force-killing worker idx={i} pid={w.pid}')
                    try:
                        w.kill()
                        w.join(timeout=5)
                    except Exception as e:
                        logger.error(f'Failed to kill idx={i}: {e}')
            pending.clear()
            break

        for idx, worker in workers:
            if idx not in pending:
                continue

            if not worker.is_alive():
                exit_code = worker.exitcode
                logger.info(
                    f'Worker idx={idx} pid={worker.pid} is no longer '
                    f'alive (exitcode={exit_code}). Joining...'
                )
                worker.join(timeout=5)
                logger.info(
                    f'Worker idx={idx} pid={worker.pid} joined '
                    f'(is_alive={worker.is_alive()})'
                )
                pending.remove(idx)
                continue

            if timeout_s is not None and (now - start_times[idx]) > timeout_s:
                logger.error(
                    'Timeout error when trying to generate an active '
                    f'configuration idx={idx}. Terminating pid={worker.pid}'
                )
                worker.terminate()
                logger.info(
                    f'Sent SIGTERM to idx={idx} pid={worker.pid}. '
                    'Calling join(timeout=5)...'
                )
                worker.join(timeout=5)
                logger.info(
                    f'join(timeout=5) returned for idx={idx} '
                    f'pid={worker.pid}. is_alive={worker.is_alive()}'
                )
                if worker.is_alive():
                    logger.error(
                        f'Worker idx={idx} pid={worker.pid} did not '
                        'terminate gracefully; sending SIGKILL'
                    )
                    if hasattr(worker, 'kill'):
                        worker.kill()
                        logger.info(
                            f'Sent SIGKILL to idx={idx} pid={worker.pid}. '
                            'Calling join(timeout=10)...'
                        )
                        worker.join(timeout=10)
                        logger.info(
                            f'join after kill returned for idx={idx} '
                            f'pid={worker.pid}. '
                            f'is_alive={worker.is_alive()}'
                        )
                    else:
                        logger.error(
                            f'Worker idx={idx} pid={worker.pid} could not '
                            'be force-killed on this Python version'
                        )
                pending.remove(idx)
                worker_results[idx] = None

        # Drain queue during the loop to prevent join/queue deadlocks
        while True:
            try:
                q_idx, q_status, q_config, q_err = result_queue.get_nowait()
                if q_status == 'ok':
                    worker_results[q_idx] = q_config
                    logger.info(
                        f'Drained result from queue: idx={q_idx} '
                        f'status={q_status}'
                    )
                else:
                    logger.error(
                        f'Drained error from queue: idx={q_idx}: ' f'{q_err}'
                    )
                    worker_results[q_idx] = None
            except Empty:
                break

        time.sleep(0.2)

    logger.info('Worker poll loop exited. Performing final queue drain...')

    # Final queue drain (pick up any remaining results)
    drain_count = 0
    while True:
        try:
            idx, status, config, err = result_queue.get_nowait()
        except Empty:
            break

        drain_count += 1
        if status == 'ok':
            worker_results[idx] = config
            logger.info(f'Final drain: idx={idx} status={status}')
        else:
            logger.error(f'Final drain: exception for idx={idx}: \n{err}')
            worker_results[idx] = None

    logger.info(
        f'Queue drain complete. Got {drain_count} item(s). '
        f'worker_results keys={sorted(worker_results.keys())}'
    )

    for idx in range(n_configs):
        configs.append(worker_results.get(idx, None))

    n_succeeded = len(configs)
    n_failed = n_configs - n_succeeded
    if n_failed > 0:
        logger.warning(
            f'{n_failed}/{n_configs} active learning workers failed '
            'or timed out'
        )
    if n_succeeded == 0:
        logger.error(
            'All active learning workers failed or timed out; '
            'no new configurations generated this iteration'
        )
    else:
        logger.info(
            f'Collected {n_succeeded}/{n_configs} active configurations'
        )

    if (
        'method_name' in kwargs
        and len(configs) > 0
        and configs.has_a_none_energy
    ):
        for config in configs:
            if config.energy.true is None:
                config.single_point(
                    kwargs['method_name'], n_cores=Config.n_cores
                )

    if (
        kwargs['inherit_metad_bias'] is True
        and kwargs['iteration'] >= kwargs['bias_start_iter']
    ):
        _generate_inheritable_metad_bias(n_configs=n_configs, kwargs=kwargs)

    mlp.training_data += configs

    os.makedirs('datasets', exist_ok=True)
    mlp.training_data.save(
        f'datasets/' f'dataset_after_iter_{kwargs["iteration"]}.npz'
    )

    if kwargs.get('keep_al_trajs') is True:
        for traj_id in range(n_configs):
            traj_name = f'trajectory_{traj_id}.traj'
            xyz_name = (
                f'al_trajectories/traj_iter{kwargs["iteration"]}_{traj_id}.xyz'
            )
            if not os.path.exists(traj_name):
                logger.warning(
                    f'Trajectory file not found for idx={traj_id}; skipping save'
                )
                continue

            _save_ase_traj_as_xyz(traj_name, xyz_name)
            os.remove(traj_name)

    return None


def _gen_active_config(
    config: 'mlptrain.Configuration',
    mlp: 'mlptrain.potentials._base.MLPotential',
    selector: 'mlptrain.training.selection.SelectionMethod',
    n_cores: int,
    max_time: float,
    method_name: str,
    **kwargs,
) -> Optional['mlptrain.Configuration']:
    """
    Generate a configuration based on 'active learning', by running MLP-MD
    until a configuration that satisfies the selection_method is found.
    This function is recursively called until a configuration is generated or
    max_time is exceeded.

    --------------------------------------------------------------------------
    Arguments:

        config: Starting configuration from which molecular dynamics is started

        mlp: Machine learnt potential from the previous active learning
             iteration

        selector: Method used to choose whether to add a given configuration to
                  the training set

        max_time: (float) Upper time limit for recursive molecular dynamics

        method_name: (str) Name of the method which we try to fit our MLP to

    Keyword Arguments:

        n_calls: (int) Number of times this function has been called

        curr_time: (float) Total time for which recursive molecular dynamics
                           has been run

        extra_time: (float) Some extra time to run initially e.g. as the
                    MLP is already likely to get to e.g. 100 fs, so run
                    that initially

        bias: Bias to add during the MD simulations, useful for exploring
              under-explored regions in the dynamics

        inherit_metad_bias: (bool) If True metadynamics bias is inherited from
                            a previous iteration to the next during active
                            learning

        bias_start_iter: (int) Iteration index at which the bias starts to be
                         applied. If the bias is PlumedBias, then UPPER_WALLS
                         and LOWER_WALLS are still applied from iteration 0

        iteration: (int) Index of the current active learning iteration

        idx: (int) Index of the current simulation

    Returns:

        (mlptrain.Configuration): Configuration which is added to the training
                                  dataset for the next iteration of active
                                  learning
    """
    curr_time = 0.0 if 'curr_time' not in kwargs else kwargs.pop('curr_time')
    extra_time = (
        0.0 if 'extra_time' not in kwargs else kwargs.pop('extra_time')
    )
    n_calls = 0 if 'n_calls' not in kwargs else kwargs.pop('n_calls')

    temp = 300.0 if 'temp' not in kwargs else kwargs.pop('temp')
    i_temp = (
        temp
        if 'init_active_temp' not in kwargs
        else kwargs.pop('init_active_temp')
    )

    pbc = False if 'pbc' not in kwargs else kwargs.pop('pbc')
    box_size = None if 'box_size' not in kwargs else kwargs.pop('box_size')

    if extra_time > 0:
        logger.info(f'Running an extra {extra_time:.1f} fs of MD')

    md_time = 2 + n_calls**3 + float(extra_time)

    if (
        kwargs['inherit_metad_bias'] is True
        and kwargs['iteration'] >= kwargs['bias_start_iter']
    ):
        kwargs = _modify_kwargs_for_metad_bias_inheritance(kwargs)

    if pbc:
        config.box = Box(box_size)

    if kwargs['md_program'].lower() == 'openmm':
        traj = run_mlp_md_openmm(
            config,
            mlp=mlp,
            temp=temp if curr_time > 0 else i_temp,
            dt=0.5,
            interval=int(max(1, 2 * md_time // selector.n_backtrack)),
            fs=md_time,
            n_cores=1,
            **kwargs,
        )
    else:
        traj = run_mlp_md(
            config,
            mlp=mlp,
            temp=temp if curr_time > 0 else i_temp,
            dt=0.5,
            interval=int(max(1, 2 * md_time // selector.n_backtrack)),
            fs=md_time,
            n_cores=1,
            **kwargs,
        )

    if traj is None:
        logger.warning('Skipping active learning step due to MD timeout.')
        return None

    traj.t0 = curr_time  # Increment the initial time (t0)

    for frame in traj:
        frame.box = Box([100, 100, 100])
    # Evaluate the selector on the final frame
    selector(traj.final_frame, mlp, method_name=method_name, n_cores=n_cores)

    if selector.select:
        if selector.check:
            logger.info(
                'currently applying distance selector,'
                'to avoid un-physical structures,'
                'do backtracking in the trajectory to'
                'find the first configuration in '
                '{selector.n_backtrack} steps recognised as outlier'
            )

            stride = max(1, len(traj) // selector.n_backtrack)

            back_traj = ConfigurationSet()
            for i in reversed(traj[::stride]):
                back_traj.append(i)

            for i, frame in enumerate(back_traj):
                logger.info(
                    f'Starting to check {i} th configuration'
                    'to determine whether it is the first'
                    'configurations selected by the distance selector'
                )
                selector(frame, mlp, method_name=method_name, n_cores=n_cores)
                if selector.select is False:
                    logger.info(f'Selecting {i-1} th configuration.')
                    frame = back_traj[i - 1]
                    break
        else:
            frame = traj.final_frame

        if frame.energy.true is None:
            frame.single_point(method_name, n_cores=n_cores)

        return frame

    if selector.too_large:
        logger.warning(
            'Backtracking in the trajectory to find a suitable '
            f'configuration in {selector.n_backtrack} steps'
        )
        stride = max(1, len(traj) // selector.n_backtrack)

        for frame in reversed(traj[::stride]):
            selector(frame, mlp, method_name=method_name, n_cores=n_cores)

            if selector.select:
                if frame.energy.true is None:
                    frame.single_point(method_name, n_cores=n_cores)

                return frame

        logger.error('Failed to backtrack to a suitable configuration')
        return None

    if curr_time + md_time > max_time:
        logger.info(f'Reached the maximum time {max_time} fs, returning None')
        return None

    # Increment t_0 to the new time
    curr_time += md_time

    # If the prediction is within the threshold then call this function again
    return _gen_active_config(
        config,
        mlp,
        selector,
        n_cores,
        max_time,
        method_name,
        temp=temp,
        curr_time=curr_time,
        n_calls=n_calls + 1,
        **kwargs,
    )


def _set_init_training_configs(
    mlp: 'mlptrain.potentials._base.MLPotential',
    init_configs: 'mlptrain.ConfigurationSet',
    method_name: str,
) -> None:
    """Set some initial training configurations"""

    if len(init_configs) == 0:
        raise ValueError(
            'Cannot set initial training configurations with a '
            'set of size 0'
        )

    if not all(cfg.energy.true is not None for cfg in init_configs):
        logger.info(
            f'Initialised with {len(init_configs)} configurations '
            f'all with defined energy'
        )
        init_configs.single_point(method=method_name)

    mlp.training_data += init_configs

    return None


def _gen_and_set_init_training_configs(
    mlp: 'mlptrain.potentials._base.MLPotential', method_name: str, num: int
) -> None:
    """
    Generate a set of initial configurations for a system, if init_configs
    is undefined. Otherwise ensure all the true energies and forces are defined
    """
    if len(mlp.training_data) >= num:
        logger.warning('MLP had sufficient training data')
        return None

    # Initial configurations are not defined, so make some - will use random
    # with the largest maximum distance between molecules possible
    max_vdw = max(atom.vdw_radius for atom in mlp.system.atoms)
    ideal_dist = 2 * max_vdw - 0.5  # Desired minimum distance in Å

    # Reduce the distance until there is a probability at least 0.1 that a
    # random configuration can be generated with that distance threshold
    p_acc, dist = 0, ideal_dist + 0.2

    while p_acc < 0.1:
        n_generated_configs = 0
        dist -= 0.2  # Reduce the minimum distance requirement

        for _ in range(10):
            try:
                _ = mlp.system.random_configuration(min_dist=dist)
                n_generated_configs += 1

            except RuntimeError:
                continue

        p_acc = n_generated_configs / 10
        logger.info(
            f'Generated configurations with p={p_acc:.2f} with a '
            f'minimum distance of {dist:.2f}'
        )

    # Generate the initial configurations
    init_configs = ConfigurationSet()
    while len(init_configs) < num:
        try:
            config = mlp.system.random_configuration(
                min_dist=dist, with_intra=True
            )
            config.box = Box([100, 100, 100])
            init_configs.append(config)

        except RuntimeError:
            continue

    logger.info(f'Added {num} configurations with min dist = {dist:.3f} Å')
    init_configs.single_point(method_name)
    mlp.training_data += init_configs
    return init_configs


def _save_ase_traj_as_xyz(
    traj_name: str,
    xyz_name: str,
) -> None:
    """
    Convert ASE trajectory to xyz format.
    """

    ase_traj = ASETrajectory(traj_name, 'r')
    ase_write(xyz_name, ase_traj, 'xyz')

    return None


def _initialise_restart(
    mlp: 'mlptrain.potentials._base.MLPotential',
    restart_iter: int,
    inherit_metad_bias: bool,
) -> None:
    """Initialise initial configurations and inherited bias"""

    init_configs = ConfigurationSet()
    init_configs.load(f'datasets/dataset_after_iter_{restart_iter}.npz')
    mlp.training_data += init_configs

    if inherit_metad_bias:
        hills_path = f'accumulated_bias/bias_after_iter_{restart_iter}.dat'
        if os.path.exists(hills_path):
            shutil.copyfile(src=hills_path, dst=f'HILLS_{restart_iter}.dat')
        else:
            raise FileNotFoundError(
                'Inherited bias generated after iteration '
                f'{restart_iter} not found'
            )

    return None


def _attach_plumed_coords_to_init_configs(
    init_configs: 'mlptrain.ConfigurationSet', bias: 'mlptrain.PlumedBias'
) -> None:
    """
    Attach PLUMED collective variable values to the configurations in the
    initial training set
    """

    logger.info('Attaching PLUMED CV values to the initial training set')

    open('init_configs_driver.xyz', 'w').close()
    for config in init_configs:
        config.save_xyz('init_configs_driver.xyz', append=True)

    driver_setup = ['UNITS LENGTH=A']
    for cv in bias.cvs:
        driver_setup.extend(cv.setup)
        driver_setup.append(
            'PRINT '
            f'ARG={cv.name} '
            f'FILE=colvar_{cv.name}_driver.dat '
            'STRIDE=1'
        )

    # Remove duplicate lines
    driver_setup = list(dict.fromkeys(driver_setup))

    with open('driver_setup.dat', 'w') as f:
        for line in driver_setup:
            f.write(f'{line}\n')

    driver_process = Popen(
        [
            'plumed',
            'driver',
            '--ixyz',
            'init_configs_driver.xyz',
            '--plumed',
            'driver_setup.dat',
            '--length-units',
            'A',
        ]
    )
    driver_process.wait()

    os.remove('init_configs_driver.xyz')
    os.remove('driver_setup.dat')

    for config in init_configs:
        config.plumed_coordinates = np.zeros(bias.n_cvs)

    for i, cv in enumerate(bias.cvs):
        colvar_fname = f'colvar_{cv.name}_driver.dat'
        cv_values = np.loadtxt(colvar_fname, usecols=1)
        os.remove(colvar_fname)

        for j, config in enumerate(init_configs):
            config.plumed_coordinates[i] = cv_values[j]

    return None


def _update_init_config(
    init_config: 'mlptrain.Configuration',
    mlp: 'mlptrain.potentials._base.MLPotential',
    fix_init_config: bool,
    bias: Optional[Union['mlptrain.Bias', 'mlptrain.PlumedBias']],
    inherit_metad_bias: bool,
    bias_start_iter: int,
    iteration: int,
) -> 'mlptrain.Configuration':
    """Update initial configuration for an active learning iteration"""

    if fix_init_config:
        return init_config

    else:
        if bias is not None:
            if inherit_metad_bias and iteration >= bias_start_iter:
                _attach_inherited_bias_energies(
                    configurations=mlp.training_data,
                    iteration=iteration,
                    bias_start_iter=bias_start_iter,
                    bias=bias,
                )

                return mlp.training_data.lowest_inherited_biased_energy

            else:
                return mlp.training_data.lowest_biased_energy

        else:
            return mlp.training_data.lowest_energy


def _check_bias(
    bias: Optional[Union['mlptrain.Bias', 'mlptrain.PlumedBias']],
    temp: float,
    inherit_metad_bias: bool,
) -> None:
    """
    Check if the bias is suitable for running active learning with the
    requested parameters
    """

    _check_bias_parameters(bias, temp)

    if inherit_metad_bias:
        _check_bias_for_metad_bias_inheritance(bias)

    return None


def _check_bias_parameters(
    bias: Optional[Union['mlptrain.Bias', 'mlptrain.PlumedBias']], temp: float
) -> None:
    """
    Check if all the required parameters of the bias are set (currently
    only checks PlumedBias initialised not from a file)
    """

    if isinstance(bias, PlumedBias):
        if bias.from_file is False and bias.metadynamics is True:
            if bias.height == 0:
                logger.info(
                    'Setting the height for metadynamics active '
                    'learning to 5*k_B*T'
                )
                bias.height = 5 * ase_units.kB * temp

    return None


def _check_bias_for_metad_bias_inheritance(bias: PlumedBias) -> None:
    """
    Check if the bias is suitable to inherit metadynamics bias during
    active learning
    """

    if not isinstance(bias, PlumedBias):
        raise TypeError(
            'Metadynamics bias can only be inherited when using PlumedBias'
        )

    if bias.from_file:
        raise ValueError(
            'Metadynamics bias cannot be inherited using '
            'PlumedBias from a file'
        )

    return None


def _remove_bias_potential(
    bias: mlptrain.Bias | PlumedBias | None = None,
) -> PlumedBias | None:
    """
    Remove bias potential from a bias, except LOWER_WALLS and UPPER_WALLS
    when the bias is PlumedBias
    """

    if isinstance(bias, PlumedBias):
        bias.strip()

    else:
        bias = None

    return bias


def _modify_kwargs_for_metad_bias_inheritance(kwargs: dict) -> dict:
    """
    Modify keyword arguments to enable metadynamics bias inheritance for
    active learning
    """

    hills_fname = f'HILLS_{kwargs["iteration"]}_{kwargs["idx"]}.dat'

    if kwargs['iteration'] > kwargs['bias_start_iter']:
        previous_hills_fname = f'HILLS_{kwargs["iteration"]-1}.dat'

        # Overwrites hills_fname when it is present during recursive MD
        shutil.copyfile(src=previous_hills_fname, dst=hills_fname)

        kwargs['copied_substrings'] = [hills_fname]
        kwargs['load_metad_bias'] = True

    kwargs['kept_substrings'] = [hills_fname]

    return kwargs


def _is_valid_hills_line(fields: List[str], expected_ncols=None) -> bool:
    """Check that a HILLS data line contains only valid numeric values.

    Returns False if any field is NaN, Inf, or non-numeric, or if the
    line has an unexpected number of columns.
    """
    if expected_ncols is not None and len(fields) != expected_ncols:
        return False

    for val in fields:
        try:
            f = float(val)
            if not np.isfinite(f):
                return False
        except ValueError:
            return False

    return True


def _generate_inheritable_metad_bias(n_configs: int, kwargs: dict) -> None:
    """
    Generate files containing metadynamics bias to be inherited in the next
    active learning iteration
    """

    iteration = kwargs['iteration']
    bias_start_iter = kwargs['bias_start_iter']

    hills_files = [f'HILLS_{iteration}_{idx}.dat' for idx in range(n_configs)]
    existing_hills_files = [
        fname for fname in hills_files if os.path.exists(fname)
    ]

    def _hills_has_data(path: str) -> bool:
        with open(path, 'r') as hills_file:
            for line in hills_file:
                if line.strip() and not line.startswith('#'):
                    return True
        return False

    valid_hills_files = [
        fname for fname in existing_hills_files if _hills_has_data(fname)
    ]
    missing_hills_files = sorted(set(hills_files) - set(existing_hills_files))
    empty_hills_files = [
        fname
        for fname in existing_hills_files
        if fname not in valid_hills_files
    ]

    if missing_hills_files:
        logger.warning(
            'Missing HILLS files detected for metadynamics bias '
            f'inheritance: {missing_hills_files}'
        )

    if empty_hills_files:
        logger.warning(
            'Empty HILLS files detected for metadynamics bias '
            f'inheritance (skipping): {empty_hills_files}'
        )

    if len(valid_hills_files) == 0:
        logger.error(
            'No non-empty HILLS files were found for generating '
            'inheritable metadynamics bias'
        )
        # Throw exception here?
        return None

    _generate_inheritable_metad_bias_hills(
        n_configs=len(valid_hills_files),
        hills_files=valid_hills_files,
        iteration=iteration,
        bias_start_iter=bias_start_iter,
    )

    return None


def _generate_inheritable_metad_bias_hills(
    n_configs: int,
    hills_files: List[str],
    iteration: int,
    bias_start_iter: int,
) -> None:
    """
    Generate HILLS_{iteration}.dat file containing metadynamics bias to be
    inherited in the next active learning iteration {iteration+1}
    """

    logger.info('Generating metadynamics bias HILLS file to inherit from')

    if iteration == bias_start_iter:
        open(f'HILLS_{iteration-1}.dat', 'w').close()

    shutil.move(src=f'HILLS_{iteration-1}.dat', dst=f'HILLS_{iteration}.dat')

    # Remove inherited bias from files containing new bias
    for fname in hills_files:
        with open(fname, 'r') as f:
            f_lines = f.readlines()

        if len(f_lines) == 0:
            continue

        prev_line = '#!'
        n_lines_in_header = 0
        second_header_first_index = 0
        for i, line in enumerate(f_lines):
            if line.startswith('#!') and not prev_line.startswith('#!'):
                second_header_first_index = i
                break
            elif line.startswith('#!'):
                n_lines_in_header += 1
            prev_line = line

        with open(fname, 'w') as f:
            # No new gaussians deposited
            if (
                second_header_first_index == 0
                and os.path.getsize(f'HILLS_{iteration}.dat') != 0
            ):
                pass

            else:
                for line in f_lines[second_header_first_index:]:
                    f.write(line)

    for idx, fname in enumerate(hills_files):
        with open(fname, 'r') as f:
            f_lines = f.readlines()

        if len(f_lines) == 0:
            os.remove(fname)
            continue

        # In some cases PLUMED fails to fully print the last line
        # Therefore, the number of columns is compared to the previous line
        if len(f_lines[-1].split()) != len(f_lines[-2].split()):
            logger.warning(
                f'Truncated last line detected in {fname}; ' 'removing it'
            )
            f_lines.pop()

        height_column_index = f_lines[0].split().index('height') - 2
        with open(f'HILLS_{iteration}.dat', 'a') as final_hills_file:
            # Attach the header to the final file if it's empty
            if os.path.getsize(f'HILLS_{iteration}.dat') == 0:
                for i in range(n_lines_in_header):
                    final_hills_file.write(f_lines[i])

            # Remove headers from contributing files
            for _ in range(n_lines_in_header):
                f_lines.pop(0)

            n_skipped = 0
            for line in f_lines:
                line_list = line.split()

                # Validate: skip lines with wrong column count,
                # non-numeric values, or NaN/Inf
                if not _is_valid_hills_line(line_list, expected_ncols=None):
                    n_skipped += 1
                    continue

                height = float(line_list[height_column_index])
                line_list[height_column_index] = f'{height / n_configs:.9f}'

                separator = '   '
                line = separator.join(line_list)

                final_hills_file.write(f'{line}\n')

            if n_skipped > 0:
                logger.warning(
                    f'Skipped {n_skipped} invalid/NaN line(s) ' f'in {fname}'
                )

        os.remove(fname)

    os.makedirs('accumulated_bias', exist_ok=True)
    shutil.copyfile(
        src=f'HILLS_{iteration}.dat',
        dst=f'accumulated_bias/bias_after_iter_{iteration}.dat',
    )

    return None


def _attach_inherited_bias_energies(
    configurations: 'mlptrain.ConfigurationSet',
    iteration: int,
    bias_start_iter: int,
    bias: 'mlptrain.PlumedBias',
) -> None:
    """
    Attach inherited metadynamics bias energies from the previous active
    learning iteration to the configurations
    """

    logger.info(
        'Attaching inherited bias energies to the whole training ' 'data set'
    )

    if iteration == bias_start_iter:
        for config in configurations:
            config.energy.inherited_bias = 0

    else:
        if os.path.getsize(f'HILLS_{iteration-1}.dat') == 0:
            for config in configurations:
                config.energy.inherited_bias = 0

            return None

        else:
            grid_ok = _generate_grid_from_hills(
                configurations=configurations, iteration=iteration, bias=bias
            )
            if not grid_ok:
                logger.warning(
                    'Failed to generate bias grid from HILLS file. '
                    'Falling back to zero inherited bias for all '
                    'configurations this iteration.'
                )
                for config in configurations:
                    config.energy.inherited_bias = 0

                return None

        cvs_cols = range(0, bias.n_metad_cvs)
        cvs_grid = np.loadtxt(
            f'bias_grid_{iteration-1}.dat', usecols=cvs_cols, ndmin=2
        )
        cvs_grid = np.flip(cvs_grid, axis=1)

        bias_grid = np.loadtxt(
            f'bias_grid_{iteration-1}.dat', usecols=bias.n_metad_cvs
        )
        bias_grid = -bias_grid

        header = []
        with open(f'bias_grid_{iteration-1}.dat', 'r') as f:
            for line in f:
                if line.startswith('#!'):
                    header.append(line)
                else:
                    break

        n_bins = []
        for cv in bias.metad_cvs:
            for line in header:
                if line.startswith(f'#! SET nbins_{cv.name}'):
                    n_bins.append(int(line.split()[-1]))
        n_bins.reverse()

        metad_cv_idxs = [bias.cvs.index(cv) for cv in bias.metad_cvs]
        metad_cv_idxs.reverse()

        for config in configurations:
            start_idxs = [0]
            block_width = np.prod(n_bins)
            for i, cv in enumerate(bias.metad_cvs):
                end_idx = start_idxs[i] + block_width

                idx = np.searchsorted(
                    a=cvs_grid[start_idxs[i] : end_idx, i],
                    v=config.plumed_coordinates[metad_cv_idxs[i]],
                    side='right',
                )
                start_idx = start_idxs[i] + idx
                start_idxs.append(start_idx)

                block_width = int(block_width / n_bins[i])

                if start_idx == end_idx:
                    raise IndexError(
                        f'CV {cv.name} value lies at the edge or '
                        f'outside of the grid for at least one '
                        f'of the configurations in the training '
                        f'set.'
                    )

            config.energy.inherited_bias = bias_grid[start_idxs[-1]]

        os.remove(f'bias_grid_{iteration-1}.dat')

    return None


def _generate_grid_from_hills(
    configurations: 'mlptrain.ConfigurationSet',
    iteration: int,
    bias: 'mlptrain.PlumedBias',
) -> bool:
    """
    Generate bias_grid_{iteration-1}.dat from HILLS_{iteration-1}.dat

    Returns:
        True if the grid file was generated successfully, False otherwise.
    """

    min_params, max_params = [], []
    metad_cv_idxs = [bias.cvs.index(cv) for cv in bias.metad_cvs]

    for j in metad_cv_idxs:
        min_value = np.min(configurations.plumed_coordinates[:, j])
        max_value = np.max(configurations.plumed_coordinates[:, j])

        difference = max_value - min_value
        extension_coefficient = 0.2
        min_params.append(min_value - difference * extension_coefficient)
        max_params.append(max_value + difference * extension_coefficient)

    bin_widths = [(width / 5) for width in bias.width]
    n_bins = [
        int((max_params[i] - min_params[i]) / bin_widths[i])
        for i in range(bias.n_metad_cvs)
    ]
    n_bins = [bins if bins != 0 else 1 for bins in n_bins]

    bin_sequence = ','.join(str(bins) for bins in n_bins)
    min_sequence = ','.join(str(param) for param in min_params)
    max_sequence = ','.join(str(param) for param in max_params)

    hills_path = f'HILLS_{iteration-1}.dat'
    grid_path = f'bias_grid_{iteration-1}.dat'

    sum_hills_process = Popen(
        [
            'plumed',
            'sum_hills',
            '--negbias',
            '--hills',
            hills_path,
            '--outfile',
            grid_path,
            '--bin',
            bin_sequence,
            '--min',
            min_sequence,
            '--max',
            max_sequence,
        ]
    )
    rc = sum_hills_process.wait()

    if rc != 0:
        logger.error(
            f'plumed sum_hills failed with exit code {rc}. '
            f'The HILLS file ({hills_path}) may contain corrupt '
            'data (e.g. NaN from a timed-out trajectory).'
        )
        return False

    if not os.path.exists(grid_path) or os.path.getsize(grid_path) == 0:
        logger.error(
            f'plumed sum_hills did not produce a valid output '
            f'file ({grid_path}). The HILLS file ({hills_path}) '
            'may be corrupt.'
        )
        return False

    return True


def _remove_last_inherited_metad_bias_file(max_active_iters: int) -> None:
    """Remove the last inherited metadynamics bias file"""

    for iteration in range(max_active_iters):
        fname = f'HILLS_{iteration}.dat'

        if os.path.exists(fname):
            os.remove(fname)
            break

    return None
