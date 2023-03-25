from copy import deepcopy
from typing import Optional, Union
from multiprocessing import Pool
from mlptrain.config import Config
from mlptrain.sampling.md import run_mlp_md
from mlptrain.sampling.plumed import PlumedBias
from mlptrain.training.selection import SelectionMethod, AbsDiffE
from mlptrain.configurations import ConfigurationSet
from mlptrain.log import logger


def train(mlp:                 'mlptrain.potentials._base.MLPotential',
          method_name:         str,
          selection_method:    SelectionMethod = AbsDiffE(),
          max_active_time:     float = 1000,
          n_configs_iter:      int = 10,
          temp:                float = 300.0,
          max_e_threshold:     Optional[float] = None,
          max_active_iters:    int = 50,
          n_init_configs:      int = 10,
          init_configs:        Optional['mlptrain.ConfigurationSet'] = None,
          fix_init_config:     bool = False,
          bbond_energy:        Optional[dict] = None,
          fbond_energy:        Optional[dict] = None,
          init_active_temp:    Optional[float] = None,
          min_active_iters:    int = 1,
          bias_start_iter:     int = 0,
          inherit_metad_bias:  bool = False,
          bias:                Union['mlptrain.sampling.Bias',
                                     'mlptrain.sampling.PlumedBias'] = None
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
                         configuration for the active learning loop, if
                         False then the minimum energy structure is used.
                         Useful for TS learning, where dynamics should be
                         propagated from a saddle point not the minimum

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
                         applied

        inherit_metad_bias: (bool) If True metadynamics bias is inherited from
                            a previous iteration to the next during active
                            learning

        bias: Bias to add during the MD simulations, useful for exploring
              under-explored regions in the dynamics
    """
    if init_configs is None:
        init_config = mlp.system.configuration
        _gen_and_set_init_training_configs(mlp,
                                           method_name=method_name,
                                           num=n_init_configs)
    else:
        init_config = init_configs[0]
        _set_init_training_configs(mlp, init_configs,
                                   method_name=method_name)

    if mlp.requires_atomic_energies:
        mlp.set_atomic_energies(method_name=method_name)

    mlp.train()

    # Run the active learning loop, running iterative GAP-MD
    for iteration in range(max_active_iters):

        curr_n_train = mlp.n_train

        _add_active_configs(mlp,
                            init_config=(init_config if fix_init_config
                                         else mlp.training_data.lowest_energy),
                            selection_method=selection_method,
                            n_configs=n_configs_iter,
                            method_name=method_name,
                            temp=temp,
                            max_time=max_active_time,
                            bbond_energy=bbond_energy,
                            fbond_energy=fbond_energy,
                            init_temp=init_active_temp,
                            extra_time=mlp.training_data.t_min(-n_configs_iter),
                            bias=bias)

        # Active learning finds no configurations,,
        if mlp.n_train == curr_n_train and iteration >= min_active_iters:
            logger.info('No AL configurations found. Final dataset size '
                        f'= {curr_n_train} Active learning = DONE')
            break

        # If required, remove high-lying energy configurations from the data
        if max_e_threshold is not None:
            mlp.training_data.remove_above_e(max_e_threshold)

        if mlp.training_data.has_a_none_energy:
            mlp.training_data.remove_none_energy()

        mlp.train()

    return None


def _add_active_configs(mlp,
                        init_config,
                        selection_method,
                        n_configs=10,
                        **kwargs
                        ) -> None:
    """
    Add a number (n_configs) of configurations to the current training data
    based on active learning selection of MLP-MD generated configurations
    """
    if Config.n_cores > n_configs and Config.n_cores % n_configs != 0:
        raise NotImplementedError('Active learning is only implemented using '
                                  'an multiple of the number n_configs_iter. '
                                  f'Please use n*{n_configs} cores.')

    n_processes = min(n_configs, Config.n_cores)
    n_cores_pp = max(Config.n_cores // n_configs, 1)
    logger.info('Searching for "active" configurations with '
                f'{n_processes} processes using {n_cores_pp} cores / process')

    configs = ConfigurationSet()

    with Pool(processes=n_processes) as pool:

        results = [pool.apply_async(_gen_active_config,
                                    args=(init_config.copy(),
                                          mlp.copy(),
                                          selection_method.copy(),
                                          n_cores_pp),
                                    kwds=deepcopy(kwargs))
                   for _ in range(n_configs)]

        for result in results:

            try:
                configs.append(result.get(timeout=None))

            # Lots of different exceptions can be raised when trying to
            # generate an active config, continue regardless..
            except Exception as err:
                logger.error(f'Raised an exception in selection: \n{err}')
                continue

    if 'method_name' in kwargs and configs.has_a_none_energy:
        configs.single_point(method=kwargs.get('method_name'))

    mlp.training_data += configs
    return None


def _gen_active_config(config:      'mlptrain.Configuration',
                       mlp:         'mlptrain.potentials._base.MLPotential',
                       selector:    'mlptrain.training.selection.SelectionMethod',
                       n_cores:     int,
                       max_time:    float,
                       method_name: str,
                       **kwargs
                       ) -> Optional['mlptrain.Configuration']:
    """
    Generate a configuration based on 'active learning', by running MLP-MD
    until a configuration that satisfies the selection_method is found.
    This function is recursively called until a configuration is generated or
    max_time is exceeded

    --------------------------------------------------------------------------
    Arguments
        config:

        mlp:

        selector:

        max_time: (float)

        method_name: (str)


    Keyword Arguments:
        n_calls: (int) Number of times this function has been called

        curr_time: (float)


        extra_time: (float) Some extra time to run initially e.g. as the
                    MLP is already likely to get to e.g. 100 fs, so run
                    that initially

    Returns:
        (mlptrain.Configuration):
    """
    curr_time = 0. if 'curr_time' not in kwargs else kwargs.pop('curr_time')
    extra_time = 0. if 'extra_time' not in kwargs else kwargs.pop('extra_time')
    n_calls = 0 if 'n_calls' not in kwargs else kwargs.pop('n_calls')

    temp = 300. if 'temp' not in kwargs else kwargs.pop('temp')
    i_temp = temp if 'init_active_temp' not in kwargs else kwargs.pop('init_active_temp')

    if extra_time > 0:
        logger.info(f'Running an extra {extra_time:.1f} fs of MD')

    md_time = 2 + n_calls**3 + float(extra_time)

    traj = run_mlp_md(config,
                      mlp=mlp,
                      temp=temp if curr_time > 0 else i_temp,
                      dt=0.5,
                      interval=max(1, 2*md_time//selector.n_backtrack),
                      fs=md_time,
                      n_cores=1,
                      **kwargs)

    traj.t0 = curr_time  # Increment the initial time (t0)

    if 'bias' in kwargs and kwargs['bias'] is not None:
        for frame in traj:
            frame.energy.predicted -= kwargs['bias'](frame.ase_atoms)

    # Evaluate the selector on the final frame
    selector(traj.final_frame, mlp, method_name=method_name, n_cores=n_cores)

    if selector.select:
        if traj.final_frame.energy.true is None:
            traj.final_frame.single_point(method_name)

        return traj.final_frame

    if selector.too_large:

        logger.warning('Backtracking in the trajectory to find a suitable '
                       f'configuration in {selector.n_backtrack} steps')
        stride = max(1, len(traj)//selector.n_backtrack)

        for frame in reversed(traj[::stride]):
            selector(frame, mlp, method_name=method_name, n_cores=n_cores)

            if selector.select:
                return frame

        logger.error('Failed to backtrack to a suitable configuration')
        return frame

    if curr_time + md_time > max_time:
        logger.info(f'Reached the maximum time {max_time} fs, returning None')
        return None

    # Increment t_0 to the new time
    curr_time += md_time

    # If the prediction is within the threshold then call this function again
    return _gen_active_config(config, mlp, selector, n_cores, max_time, method_name,
                              temp=temp,
                              curr_time=curr_time,
                              n_calls=n_calls+1,
                              **kwargs)


def _set_init_training_configs(mlp, init_configs, method_name) -> None:
    """Set some initial training configurations"""

    if len(init_configs) == 0:
        raise ValueError('Cannot set initial training configurations with a '
                         'set of size 0')

    if not all(cfg.energy.true is not None for cfg in init_configs):
        logger.info(f'Initialised with {len(init_configs)} configurations '
                    f'all with defined energy')
        init_configs.single_point(method=method_name)

    mlp.training_data += init_configs

    return None


def _gen_and_set_init_training_configs(mlp, method_name, num) -> None:
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
        dist -= 0.2                # Reduce the minimum distance requirement

        for _ in range(10):
            try:
                _ = mlp.system.random_configuration(min_dist=dist)
                n_generated_configs += 1

            except RuntimeError:
                continue

        p_acc = n_generated_configs / 10
        logger.info(f'Generated configurations with p={p_acc:.2f} with a '
                    f'minimum distance of {dist:.2f}')

    # Generate the initial configurations
    init_configs = ConfigurationSet()
    while len(init_configs) < num:
        try:
            config = mlp.system.random_configuration(min_dist=dist,
                                                     with_intra=True)
            init_configs.append(config)

        except RuntimeError:
            continue

    logger.info(f'Added {num} configurations with min dist = {dist:.3f} Å')
    init_configs.single_point(method_name)
    mlp.training_data += init_configs
    return None
