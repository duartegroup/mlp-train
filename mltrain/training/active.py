from typing import Optional
from subprocess import Popen, PIPE
from multiprocessing import Pool
from mltrain.config import Config
from mltrain.training.selection import SelectionMethod, AbsDiffE
from mltrain.configurations import ConfigurationSet
from mltrain.log import logger

"""
if active_e_thresh is None:
    if active_method.lower() == 'diff':
        #                 1 kcal mol-1 molecule-1
        active_e_thresh = 0.043363 * len(mlp.system.molecules)

    if active_method.lower() == 'gp_var':
        # Threshold for maximum per-atom GP variance (eV atom^-1)
        active_e_thresh = 1E-5
"""


def train(mlp:               'mltrain.potentials.MLPotential',
          method_name:       str,
          selection_method:  SelectionMethod = AbsDiffE,
          max_active_time:   float = 1000,
          n_configs_iter:    int = 10,
          temp:              float = 300.0,
          max_e_threshold:   Optional[float] = None,
          max_active_iters:  int = 50,
          n_init_configs:    int = 10,
          init_configs:      Optional['mltrain.ConfigurationSet'] = None,
          fix_init_config:   bool = False,
          bbond_energy:      Optional[dict] = None,
          fbond_energy:      Optional[dict] = None,
          init_active_temp:  Optional[float] = None,
          min_active_iters:  int = 1) -> None:
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

    Keyword Arguments:
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
    """
    if init_configs is None:
        _gen_init_training_configs(mlp,
                                   method_name=method_name,
                                   num=n_init_configs)
    else:
        _set_init_training_configs(mlp, init_configs,
                                   method_name=method_name)

    mlp.train()

    # Run the active learning loop, running iterative GAP-MD
    for iteration in range(max_active_iters):

        curr_n_configs = len(mlp.training_data)

        _add_active_configs(mlp,
                            init_config=(mlp.training_data[0] if fix_init_config
                                         else mlp.training_data.lowest_energy),
                            selection_method=selection_method,
                            n_configs=n_configs_iter,
                            method_name=method_name,
                            temp=temp,
                            max_time=max_active_time,
                            bbond_energy=bbond_energy,
                            fbond_energy=fbond_energy,
                            init_temp=init_active_temp)

        # Active learning finds no configurations,,
        if len(mlp.training_data) == curr_n_configs and iteration > min_active_iters:
            logger.info('No AL configurations found. Final dataset size '
                        f'= {curr_n_configs} Active learning = DONE')
            break

        # If required, remove high-lying energy configuration from the data
        if max_e_threshold is not None:
            mlp.training_data.remove_above_e(max_e_threshold)

        mlp.train()

    return None


def _add_active_configs(mlp,
                        init_config,
                        selection_method,
                        n_configs=10,
                        **kwargs) -> None:
    """
    Add a number (n_configs) of configurations to the current training data
    based on active learning selection of MLP-MD generated configurations
    """
    if int(n_configs) < int(Config.n_cores):
        raise NotImplementedError('Active learning is only implemented using '
                                  'one core for each process. Please use '
                                  'n_configs >= mlt.Config.n_cores')

    configs = ConfigurationSet()
    logger.info('Searching for "active" configurations with '
                f'{Config.n_cores} processes')

    with Pool(processes=Config.n_cores) as pool:

        results = [pool.apply_async(selection_method, args=(init_config,), kwds=kwargs)
                   for _ in range(n_configs)]

        for result in results:

            try:
                configs += result.get(timeout=None)

            # Lots of different exceptions can be raised when trying to
            # generate an active config, continue regardless..
            except Exception as err:
                logger.error(f'Raised an exception in selection: \n{err}')
                continue

    mlp.training_data += configs
    return None


def _set_init_training_configs(mlp, init_configs, method_name) -> None:
    """Set some initial training configurations"""

    if not all(cfg.energy.true is not None for cfg in init_configs):
        logger.info(f'Initialised with {len(init_configs)} configurations '
                    f'all with defined energy')
        init_configs.single_point(method_name=method_name)

    mlp.training_data = init_configs

    return None


def _gen_init_training_configs(mlp, method_name, num) -> None:
    """
    Generate a set of initial configurations for a system, if init_configs
    is undefined. Otherwise ensure all the true energies and forces are defined

    Arguments:
        mlp:
        method_name:
        num:
    """
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
    while len(mlp.training_data) < num:
        try:
            config = mlp.system.random_configuration(min_dist=dist,
                                                     with_intra=True)
            mlp.training_data += config

        except RuntimeError:
            continue

    logger.info(f'Added {num} configurations with min dist = {dist:.3f} Å')

    mlp.training_data.single_point(method_name)
    return None
