import mlptrain
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress


mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['axes.linewidth'] = 1.2


def parity_plot(
    config_set: 'mlptrain.ConfigurationSet', name: str = 'parity'
) -> None:
    """
    Plot parity plots of energies, forces and temporal differences (if present)
    otherwise the residuals over the configuration index

    ---------------------------------------------------------------------------
    Arguments:
        config_set: Set of configurations

        name:
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 7.5))

    if _all_energies_are_defined(config_set):
        _add_energy_time_plot(config_set, axis=ax[0, 0])
        _add_energy_parity_plot(config_set, axis=ax[0, 1])

    if _all_forces_are_defined(config_set):
        _add_force_component_plot(config_set, axis=ax[1, 0])
        _add_force_magnitude_plot(config_set, axis=ax[1, 1])

    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    return None


def error_histogram(
    config_set: 'mlptrain.ConfigurationSet', name: str = 'error_histogram'
) -> None:
    """
    Plot distribution of errors in energies and forces for given configuration set

    ------------------------------------------------------------------------------
    Arguments:
        config_set: Set of configurations

        name: name of the file

    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.75))

    if _all_energies_are_defined(config_set):
        _add_energy_error_histogram(config_set, axis=ax[0])

    if _all_forces_are_defined(config_set):
        _add_force_error_histogram(config_set, axis=ax[1])

    plt.tight_layout()
    plt.savefig(f'{name}.pdf')


def _all_energies_are_defined(cfgs) -> bool:
    """Are all the energies defined in a configuration set?"""
    return all(
        e is not None for e in cfgs.true_energies + cfgs.predicted_energies
    )


def _all_forces_are_defined(cfgs) -> bool:
    """Are all the forces defined in a configuration set?"""
    return cfgs.true_forces is not None and cfgs.predicted_forces is not None


def _add_energy_time_plot(config_set, axis) -> None:
    """Plot energies vs time, if undefined then use the index"""

    if config_set[0].time is None:
        xs = np.arange(0, len(config_set))
        xlabel = 'Index'

    else:
        xs = [frame.time for frame in config_set]
        xlabel = 'Time / fs'

    true_Es = np.array(config_set.true_energies)
    min_E = np.min(true_Es)

    axis.plot(
        xs,
        np.array(config_set.predicted_energies) - min_E,
        label='predicted',
        lw=2,
    )

    axis.plot(xs, true_Es - min_E, label='true', c='orange', lw=2)

    # plot the region of 'chemical accuracy' 1 kcal mol-1 = 0.043 eV
    axis.fill_between(
        xs,
        y1=true_Es - min_E - 0.043,
        y2=true_Es - min_E + 0.043,
        alpha=0.2,
        color='orange',
    )

    axis.legend()
    axis.set_xlabel(xlabel)
    axis.set_ylabel('$E - E_{min, true}$ (eV)')

    return None


def _add_energy_parity_plot(config_set, axis) -> None:
    """Plot true vs predicted energies"""
    x = np.array(config_set.true_energies)
    xs = x - np.min(x)  # Only relative energies matter

    y = np.array(config_set.predicted_energies)
    ys = y - np.min(y)

    min_e = min([np.min(xs), np.min(ys)])
    max_e = min([np.max(xs), np.max(ys)])

    axis.scatter(xs, ys, marker='o', s=20, c='white', edgecolors='blue')

    axis.plot([min_e, max_e], [min_e, max_e], c='k', lw=1.0)

    _add_r_sq_and_mad(axis, x=x, y=y, xs=xs, ys=ys, unit='meV')

    axis.set_xlim(min_e, max_e)
    axis.set_ylim(min_e, max_e)

    axis.set_xlabel('$E_{rel, true}$ (eV)')
    axis.set_ylabel('$E_{rel, predicted}$ (eV)')

    return None


def _add_force_component_plot(config_set, axis) -> None:
    """Add a parity plot of the force components"""
    cmaps = [
        plt.get_cmap('Blues'),
        plt.get_cmap('Reds'),
        plt.get_cmap('Purples'),
    ]

    # get the min and max force components in any of (x, y, z) directions for plotting
    min_true_f = min(
        [np.min(config_forces) for config_forces in config_set.true_forces]
    )

    min_pred_f = min(
        [
            np.min(config_forces)
            for config_forces in config_set.predicted_forces
        ]
    )

    max_true_f = max(
        [np.max(config_forces) for config_forces in config_set.true_forces]
    )

    max_pred_f = max(
        [
            np.max(config_forces)
            for config_forces in config_set.predicted_forces
        ]
    )

    min_f = min([min_true_f, min_pred_f])
    max_f = min([max_true_f, max_pred_f])

    for idx, k in enumerate(['x', 'y', 'z']):
        xs, ys = [], []
        for config in config_set:
            xs += config.forces.true[:, idx].tolist()
            ys += config.forces.predicted[:, idx].tolist()

        axis.hist2d(
            xs,
            ys,
            bins=100,
            label='$F_{x}$',
            cmap=cmaps[idx],
            norm=mpl.colors.LogNorm(),
        )

    axis.set_ylim(min_f, max_f)
    axis.set_xlim(min_f, max_f)

    axis.set_xlabel('$F_{true}$ (eV Å$^{-1})$')
    axis.set_ylabel('$F_{predicted}$ (eV Å$^{-1})$')

    return None


def _add_force_magnitude_plot(config_set, axis) -> None:
    """Add a parity plot of the force magnitudes"""

    x, y = [], []
    for config in config_set:
        x += np.linalg.norm(config.forces.true, axis=1).tolist()
        y += np.linalg.norm(config.forces.predicted, axis=1).tolist()

    min_f = min([np.min(x), np.min(y)])
    max_f = min([np.max(x), np.max(y)])

    axis.hist2d(
        x,
        y,
        range=[[min_f, max_f], [min_f, max_f]],
        bins=100,
        cmap=plt.get_cmap('Blues'),
        norm=mpl.colors.LogNorm(),
    )

    _add_r_sq_and_mad(axis, x=np.array(x), y=np.array(y), unit='meV Å$^{-1}$)')

    axis.set_ylim(min_f, max_f)
    axis.set_xlim(min_f, max_f)

    axis.set_xlabel('$|{\\bf{F}}|_{true}$ (eV Å$^{-1}$)')
    axis.set_ylabel('$|{\\bf{F}}|_{predicted}$ (eV Å$^{-1}$)')

    return None


def _add_energy_error_histogram(
    config_set, axis, per_atom=True, print_structures=True, N=3
) -> None:
    """Add histogram of energy errors"""

    x = np.array(config_set.true_energies)

    y = np.array(config_set.predicted_energies)

    n_atoms = len(config_set[0].atoms)

    if per_atom:
        error_abs = np.abs(x - y) * 1000 / n_atoms
    else:
        error_abs = np.abs(x - y) * 1000

    min_e = min(error_abs)
    max_e = max(error_abs)

    sns.histplot(
        error_abs, bins=30, color='blue', alpha=0.7, kde=False, ax=axis
    )

    axis.set_xlim(min_e, max_e)
    # axis.set_ylim(min_e, max_e)

    if per_atom:
        mad = _add_max_and_mad(
            axis, x=x / n_atoms, y=y / n_atoms, unit='meV atom$^{-1}$'
        )

        axis.set_xlabel('Error on energy (meV atom$^{-1}$)')
    else:
        mad = _add_max_and_mad(axis, x=x, y=y, units='meV')
        axis.set_xlabel('Error on energy (meV)')

    axis.set_ylabel('Occurence')

    if print_structures:
        data = mlptrain.ConfigurationSet()
        for i, structure in enumerate(config_set):
            if error_abs[i] >= N * mad:
                data.append(structure)

        data.save_xyz(f'structure_{N}_mad_en_error.xyz')


def _add_force_error_histogram(
    config_set, axis, print_structures=True, N=5
) -> None:
    """Add histogram of force errors"""

    x, y = [], []
    for config in config_set:
        x.append(np.linalg.norm(config.forces.true, axis=1))
        y.append(np.linalg.norm(config.forces.predicted, axis=1))

    force_errors = np.abs(np.array(y) - np.array(x)) * 1000

    force_errors_all = np.concatenate(np.abs(np.array(y) - np.array(x)) * 1000)

    min_f = min(force_errors_all)
    max_f = max(force_errors_all)

    sns.histplot(
        force_errors_all,
        bins=30,
        color='orange',
        alpha=0.7,
        kde=False,
        ax=axis,
    )

    mad = _add_max_and_mad(
        axis, x=np.array(x), y=np.array(y), unit='meV Å$^{-1}$'
    )

    axis.set_xlim(min_f, max_f)

    axis.set_xlabel('Error on $|{\\bf{F}}|$ (meV Å$^{-1}$)')

    axis.set_yscale('log')
    axis.set_ylabel('Occurence')

    if print_structures:
        data = mlptrain.ConfigurationSet()
        for i, structure in enumerate(config_set):
            if any(force_errors[i] >= N * mad):
                data.append(structure)

        data.save_xyz(f'structure_{N}_mad_f_error.xyz')


def _add_r_sq_and_mad(axis, x, y, unit, xs=None, ys=None):
    """
    Add an annotation of the correlation and MAD between the data
    -------------------------------------------------------------
    Arguments:
    xs, ys: Values shifted by minimum value, optional.
    x,y : Non-modified values

    """

    if xs is not None and ys is not None:
        slope, intercept, r, p, se = linregress(xs, ys)
        axis.annotate(
            f'$R^2$ = {r**2:.3f}\n'
            f' MAD$_{{relative}}$ = {np.mean(np.abs(xs - ys)):.3f} {unit}\n'
            f'MAD = {np.mean(np.abs(x - y)):.3f} {unit}',
            xy=(1, 0),
            xycoords='axes fraction',
            fontsize=12,
            xytext=(-5, 5),
            textcoords='offset points',
            ha='right',
            va='bottom',
        )
    else:
        slope, intercept, r, p, se = linregress(x, y)
        axis.annotate(
            f'$R^2$ = {r**2:.3f}\n'
            f'MAD = {np.mean(np.abs(x - y)):.3f} {unit}',
            xy=(1, 0),
            xycoords='axes fraction',
            fontsize=12,
            xytext=(-5, 5),
            textcoords='offset points',
            ha='right',
            va='bottom',
        )

    return None


def _add_max_and_mad(axis, x, y, unit):
    """
    Add an annotation of the MAD and maximum value between the data
    ---------------------------------------------------------------
    Arguments:
       x,y : Non-modified values
       unit: (str)
    """
    mad = np.mean(np.abs(x - y)) * 1000

    axis.annotate(
        f'MAD = {mad:.3f} {unit}\n'
        f'MAX = {np.max(np.abs(x - y))*1000:.3f} {unit}',
        xy=(1, 1),
        xycoords='axes fraction',
        fontsize=12,
        xytext=(-5, -5),
        textcoords='offset points',
        ha='right',
        va='top',
    )

    return mad
