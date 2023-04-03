import os
import io
import shutil
import numpy as np
from copy import deepcopy
from typing import Optional, Union, Dict, List
from multiprocessing import Pool
from subprocess import Popen
from ase import units as ase_units
from mlptrain.config import Config
from mlptrain.sampling import PlumedBias
from mlptrain.sampling.md import run_mlp_md
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
          constraints:         Optional[List] = None,
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
                         False then the structure with lowest biased energy
                         (true energy + bias energy) is used. Useful for
                         TS learning, where dynamics should be propagated
                         from a saddle point not the minimum

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

        inherit_metad_bias: (bool) If True metadynamics bias is inherited from
                            a previous iteration to the next during active
                            learning

        constraints: (List) List of ASE contraints to use in the dynamics
                            during active learning

        bias: Bias to add during the MD simulations, useful for exploring
              under-explored regions in the dynamics
    """

    _check_bias(bias, temp, inherit_metad_bias)

    if init_configs is None:
        init_config = mlp.system.configuration
        _gen_and_set_init_training_configs(mlp,
                                           method_name=method_name,
                                           num=n_init_configs)

    else:
        init_config = init_configs[0]
        _set_init_training_configs(mlp, init_configs,
                                   method_name=method_name)

    if isinstance(bias, PlumedBias):
        _attach_plumed_coords_to_init_configs(init_configs=mlp.training_data,
                                              bias=bias)

    if mlp.requires_atomic_energies:
        mlp.set_atomic_energies(method_name=method_name)

    mlp.train()

    # Run the active learning loop, running iterative GAP-MD
    for iteration in range(max_active_iters):

        curr_n_train = mlp.n_train

        if inherit_metad_bias and iteration >= bias_start_iter:
            _attach_inherited_bias_energies(configurations=mlp.training_data,
                                            iteration=iteration,
                                            bias_start_iter=bias_start_iter,
                                            bias=bias)

        init_config_iter = _update_init_config(init_config, mlp,
                                               fix_init_config,
                                               inherit_metad_bias)

        _add_active_configs(mlp,
                            init_config=init_config_iter,
                            selection_method=selection_method,
                            n_configs=n_configs_iter,
                            method_name=method_name,
                            temp=temp,
                            max_time=max_active_time,
                            bbond_energy=bbond_energy,
                            fbond_energy=fbond_energy,
                            init_temp=init_active_temp,
                            extra_time=mlp.training_data.t_min(-n_configs_iter),
                            constraints=constraints,
                            bias=deepcopy(bias),
                            inherit_metad_bias=inherit_metad_bias,
                            bias_start_iter=bias_start_iter,
                            iteration=iteration)

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

    if inherit_metad_bias:
        _remove_last_inherited_metad_bias_file(max_active_iters, bias)

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

    if 'bias' in kwargs and kwargs['iteration'] < kwargs['bias_start_iter']:
        logger.info(f'Iteration {kwargs["iteration"]}: the bias potential '
                    'is not applied')
        kwargs['bias'] = _remove_bias_potential(kwargs['bias'])

    configs = ConfigurationSet()
    results = []

    with Pool(processes=n_processes) as pool:

        for idx in range(n_configs):
            kwargs['idx'] = idx

            result = pool.apply_async(_gen_active_config,
                                      args=(init_config.copy(),
                                            mlp.copy(),
                                            selection_method.copy(),
                                            n_cores_pp),
                                      kwds=deepcopy(kwargs))
            results.append(result)

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

    if (kwargs['inherit_metad_bias'] is True
            and kwargs['iteration'] >= kwargs['bias_start_iter']):

        _generate_inheritable_metad_bias(n_configs, kwargs)

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
    curr_time = 0. if 'curr_time' not in kwargs else kwargs.pop('curr_time')
    extra_time = 0. if 'extra_time' not in kwargs else kwargs.pop('extra_time')
    n_calls = 0 if 'n_calls' not in kwargs else kwargs.pop('n_calls')

    temp = 300. if 'temp' not in kwargs else kwargs.pop('temp')
    i_temp = temp if 'init_active_temp' not in kwargs else kwargs.pop('init_active_temp')

    if extra_time > 0:
        logger.info(f'Running an extra {extra_time:.1f} fs of MD')

    md_time = 2 + n_calls**3 + float(extra_time)

    if (kwargs['inherit_metad_bias'] is True
            and kwargs['iteration'] >= kwargs['bias_start_iter']):

        kwargs = _modify_kwargs_for_metad_bias_inheritance(kwargs)

    traj = run_mlp_md(config,
                      mlp=mlp,
                      temp=temp if curr_time > 0 else i_temp,
                      dt=0.5,
                      interval=max(1, 2*md_time//selector.n_backtrack),
                      fs=md_time,
                      n_cores=1,
                      **kwargs)

    traj.t0 = curr_time  # Increment the initial time (t0)

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
                if traj.final_frame.energy.true is None:
                    traj.final_frame.single_point(method_name)

                return frame

        logger.error('Failed to backtrack to a suitable configuration')
        return None

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


def _attach_plumed_coords_to_init_configs(init_configs, bias) -> None:
    """Attaches PLUMED collective variable values to the configurations
    in the initial training set"""

    open('init_configs_driver.xyz', 'w').close()
    for config in init_configs:
        config.save_xyz('init_configs_driver.xyz', append=True)

    driver_setup = [f'UNITS LENGTH=A']
    for cv in bias.cvs:
        driver_setup.extend(cv.setup)
        driver_setup.append('PRINT '
                            f'ARG={cv.name} '
                            f'FILE=colvar_{cv.name}_driver.dat '
                            'STRIDE=1')

    with open('driver_setup.dat', 'w') as f:
        for line in driver_setup:
            f.write(f'{line}\n')

    driver_process = Popen(['plumed', 'driver',
                            '--ixyz', 'init_configs_driver.xyz',
                            '--plumed', 'driver_setup.dat',
                            '--length-units', 'A'])
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


def _update_init_config(init_config, mlp, fix_init_config, inherit_metad_bias
                        ) -> 'mlptrain.Configuration':
    """Updates initial configuration for an active learning iteration"""

    if fix_init_config:
        return init_config

    else:
        return mlp.training_data.lowest_biased_energy(inherited=inherit_metad_bias)


def _check_bias(bias, temp, inherit_metad_bias) -> None:
    """Checks if the bias is suitable for running active learning with the
    requested parameters"""

    _check_bias_parameters(bias, temp)

    if inherit_metad_bias:
        _check_bias_for_metad_bias_inheritance(bias)

    return None


def _check_bias_parameters(bias, temp) -> None:
    """Checks if all the required parameters of the bias are set (currently
    only checks PlumedBias initialised not from a file)"""

    if isinstance(bias, PlumedBias):

        if bias.setup is None and bias.metadynamics is True:

            # 1E9 == dummy height value
            if bias.height == 1E9:
                logger.info('Setting the height for metadynamics active '
                            'learning to 0.5*k_B*T')
                bias.height = 0.5 * ase_units.kB * temp

    return None


def _check_bias_for_metad_bias_inheritance(bias) -> None:
    """Checks if the bias is suitable to inherit metadynamics bias during
    active learning"""

    if not isinstance(bias, PlumedBias):
        raise TypeError('Metadynamics bias can only be inherited when '
                        'using PlumedBias')

    if bias.setup is not None:
        raise ValueError('Metadynamics bias cannot be inherited using '
                         'PlumedBias from a file')

    return None


def _remove_bias_potential(bias) -> Union['mlptrain.sampling.PlumedBias', None]:
    """Removes bias potential from a bias, except LOWER_WALLS and UPPER_WALLS
    when the bias is PlumedBias"""

    if isinstance(bias, PlumedBias):
        bias.strip()

    else:
        bias = None

    return bias


def _modify_kwargs_for_metad_bias_inheritance(kwargs) -> Dict:
    """Modifies keyword arguments to enable metadynamics bias inheritance for
    active learning"""

    bias = kwargs['bias']

    if bias.metad_grid_min is not None and bias.metad_grid_max is not None:

        if kwargs['iteration'] == kwargs['bias_start_iter']:
            previous_grid_fname = None

        else:
            previous_grid_fname = f'bias_grid_{kwargs["iteration"]-1}.dat'
            kwargs['copied_substrings'] = ['.xml', '.json', '.pth',
                                           previous_grid_fname]

        grid_fname = f'bias_grid_{kwargs["iteration"]}_{kwargs["idx"]}.dat'
        kwargs['kept_substrings'] = [grid_fname]

        bias._set_metad_grid_params(grid_min=bias.metad_grid_min,
                                    grid_max=bias.metad_grid_max,
                                    grid_bin=bias.metad_grid_bin,
                                    grid_wstride=bias.pace,
                                    grid_wfile=grid_fname,
                                    grid_rfile=previous_grid_fname)

    # Using HILLS instead of grids
    else:
        hills_fname = f'HILLS_{kwargs["iteration"]}_{kwargs["idx"]}.dat'

        if kwargs['iteration'] > kwargs['bias_start_iter']:
            previous_hills_fname = f'HILLS_{kwargs["iteration"]-1}.dat'

            # Overwrites hills_fname when it is present during recursive MD
            shutil.copyfile(src=previous_hills_fname, dst=hills_fname)

            kwargs['copied_substrings'] = ['.xml', '.json', '.pth', hills_fname]

        kwargs['kept_substrings'] = [hills_fname]

    if kwargs['iteration'] > kwargs['bias_start_iter']:
        kwargs['load_metad_bias'] = True

    return kwargs


def _generate_inheritable_metad_bias(n_configs, kwargs) -> None:
    """Generates files containing metadynamics bias to be inherited in the next
    active learning iteration"""

    bias = kwargs['bias']
    iteration = kwargs['iteration']
    bias_start_iter = kwargs['bias_start_iter']

    grid_files = [f'bias_grid_{iteration}_{idx}.dat' for idx in range(n_configs)]
    using_grids = all(os.path.exists(fname) for fname in grid_files)

    hills_files = [f'HILLS_{iteration}_{idx}.dat' for idx in range(n_configs)]
    using_hills = all(os.path.exists(fname) for fname in hills_files)

    if using_grids:
        _generate_inheritable_metad_bias_grid(n_configs, grid_files, bias,
                                              iteration, bias_start_iter)

    elif using_hills:
        _generate_inheritable_metad_bias_hills(n_configs, hills_files,
                                               iteration, bias_start_iter)

    else:
        logger.error('All files required for generating inheritable '
                     'metadynamics bias could not be found')

    return None


def _generate_inheritable_metad_bias_grid(n_configs, grid_files, bias,
                                          iteration, bias_start_iter) -> None:
    """Generates bias_grid_{iteration}.dat file containing metadynamics bias to
    be inherited in the next active learning iteration {iteration+1}"""

    logger.info('Generating metadynamics bias grid file to inherit from')

    if iteration > bias_start_iter:
        os.remove(f'bias_grid_{iteration-1}.dat')

    # Extract the header
    with open(f'bias_grid_{iteration}_0.dat', 'r') as f:

        header = ''
        for line in f:

            if line.startswith('#!'):
                header += line

            else:
                break

    # cv1 cv2 ... metad.bias der_cv1 der_cv2 ...
    cvs_cols = range(0, bias.n_cvs)
    cvs = np.loadtxt(f'bias_grid_{iteration}_0.dat', usecols=cvs_cols, ndmin=2)

    # data is bias and derivatives, which is going to be averaged over files
    data_cols = range(bias.n_cvs, 2 * bias.n_cvs + 1)
    n_cols = len(data_cols)
    n_rows = len(cvs)
    data = np.zeros((n_rows, n_cols))

    for fname in grid_files:
        data += np.loadtxt(fname, usecols=data_cols)
        os.remove(fname)
        os.remove(f'bck.last.{fname}')

    mean_data = data / n_configs
    final_array = np.concatenate((cvs, mean_data), axis=1)

    with open(f'bias_grid_{iteration}.dat', 'w') as f:

        for line in header:
            f.write(line)

        # Save str representation of the array in memory
        bytes_io = io.BytesIO()
        np.savetxt(fname=bytes_io, X=final_array, fmt='    %.9f')
        f.write(bytes_io.getvalue().decode())

    return None


def _generate_inheritable_metad_bias_hills(n_configs, hills_files, iteration,
                                           bias_start_iter) -> None:
    """Generates HILLS_{iteration}.dat file containing metadynamics bias to be
    inherited in the next active learning iteration {iteration+1}"""

    logger.info('Generating metadynamics bias HILLS file to inherit from')

    if iteration > bias_start_iter:

        shutil.move(src=f'HILLS_{iteration-1}.dat',
                    dst=f'HILLS_{iteration}.dat')

        # Remove inherited bias from files containing new bias
        for fname in hills_files:

            with open(fname, 'r') as f:
                f_lines = f.readlines()

            if len(f_lines) == 0:
                continue

            first_header_indices = [0, 1, 2]
            second_header_first_index = 0
            for i, line in enumerate(f_lines):
                if line.startswith('#!') and i not in first_header_indices:
                    second_header_first_index = i
                    break

            with open(fname, 'w') as f:

                # No new gaussians deposited
                if second_header_first_index == 0:
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

        has_biasf = f_lines[0].split()[-1] == 'biasf'
        height_column_index = -2 if has_biasf else -1

        with open(f'HILLS_{iteration}.dat', 'a') as final_hills_file:

            # Attach the header to the final file if it's empty
            if os.path.getsize(f'HILLS_{iteration}.dat') == 0:
                for line in f_lines[:3]:
                    final_hills_file.write(line)

            # Remove headers from contributing files
            for _ in range(3):
                f_lines.pop(0)

            for line in f_lines:
                line_list = line.split()
                height = float(line_list[height_column_index])
                line_list[height_column_index] = f'{height / n_configs:.9f}'

                separator = '   '
                line = separator.join(line_list)

                final_hills_file.write(f'{line}\n')

        os.remove(fname)

    return None


def _attach_inherited_bias_energies(configurations, iteration,
                                    bias_start_iter, bias) -> None:
    """Attaches inherited metadynamics bias energies from the previous active
    learning iteration to the configurations"""

    if iteration == bias_start_iter:
        for config in configurations:
            config.energy.inherited_bias = 0

    else:
        inheritance_using_hills = os.path.exists(f'HILLS_{iteration-1}.dat')

        if inheritance_using_hills:
            _generate_grid_from_hills(configurations, iteration, bias)

        cvs_and_bias_cols = range(0, bias.n_cvs + 1)
        cvs_and_bias = np.loadtxt(f'bias_grid_{iteration-1}.dat',
                                  usecols=cvs_and_bias_cols)

        if inheritance_using_hills:
            cvs_and_bias[:, -1] = -cvs_and_bias[:, -1]

        header = []
        with open(f'bias_grid_{iteration-1}.dat', 'r') as f:
            for line in f:
                if line.startswith('#!'):
                    header.append(line)
                else:
                    break

        n_bins = []
        for cv in bias.cvs:
            for line in header:
                if line.startswith(f'#! SET nbins_{cv.name}'):
                    n_bins.append(int(line.split()[-1]))

        for config in configurations:

            start_idxs = []
            block_width = np.prod(n_bins)
            for i, cv in enumerate(bias.n_cvs):

                end_idx = start_idxs[i] + block_width

                idx = np.searchsorted(a=cvs_and_bias[start_idxs[i]:end_idx, i],
                                      v=config.plumed_coordinates[i],
                                      side='right')
                start_idx = start_idxs[i] + idx
                start_idxs.append(start_idx)

                block_width = int(block_width / n_bins[i])

                if start_idx == end_idx:
                    raise IndexError(f'CV {cv.name} value lies at the edge or '
                                     f'outside of the grid for one of the '
                                     f'configurations in the training set. '
                                     f'Please use a larger grid')

            config.energy.inherited_bias = cvs_and_bias[start_idxs[-1], bias.n_cvs]

        if inheritance_using_hills:
            os.remove(f'bias_grid_{iteration-1}.dat')

    return None


def _generate_grid_from_hills(configurations, iteration, bias) -> None:
    """Generates bias_grid_{iteration-1}.dat from HILLS_{iteration-1}.dat"""

    min_params, max_params = [], []

    for j in range(bias.n_cvs):
        min_value = np.min(configurations.plumed_coordinates[:, j])
        max_value = np.max(configurations.plumed_coordinates[:, j])

        difference = max_value - min_value
        extension_coefficient = 0.2
        min_params.append(min_value - difference * extension_coefficient)
        max_params.append(max_value + difference * extension_coefficient)

    bin_widths = [(width / 5) for width in bias.width]
    n_bins = [(max_params[i] - min_params[i]) // bin_widths[i] for i in bias.n_cvs]
    bin_sequence = ','.join(str(bins) for bins in n_bins)

    min_sequence = ','.join(str(param) for param in min_params)
    max_sequence = ','.join(str(param) for param in max_params)

    sum_hills_process = Popen(['plumed', 'sum_hills', '--negbias',
                               '--hills', f'HILLS_{iteration-1}',
                               '--outfile', f'bias_grid_{iteration-1}.dat',
                               '--bin', bin_sequence,
                               '--min', min_sequence,
                               '--max', max_sequence])
    sum_hills_process.wait()

    return None


def _remove_last_inherited_metad_bias_file(max_active_iters, bias) -> None:
    """Removes the last inherited metadynamics bias file"""

    for iteration in range(max_active_iters):

        if bias.metad_grid_min is not None and bias.metad_grid_max is not None:
            fname = f'bias_grid_{iteration}.dat'

        else:
            fname = f'HILLS_{iteration}.dat'

        if os.path.exists(fname):
            os.remove(fname)
            break

    return None
