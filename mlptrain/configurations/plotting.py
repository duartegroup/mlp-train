import mlptrain
import numpy as np
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
    axis.set_ylabel('$E$ / eV')

    return None


def _add_energy_parity_plot(config_set, axis) -> None:
    """Plot true vs predicted energies"""
    xs = np.array(config_set.true_energies)
    xs -= np.min(xs)  # Only relative energies matter

    ys = np.array(config_set.predicted_energies)
    ys -= np.min(ys)

    min_e = min([np.min(xs), np.min(ys)])
    max_e = min([np.max(xs), np.max(ys)])

    axis.scatter(xs, ys, marker='o', s=20, c='white', edgecolors='blue')

    axis.plot([min_e, max_e], [min_e, max_e], c='k', lw=1.0)

    _add_r_sq_and_mad(axis, xs=xs, ys=ys)

    axis.set_xlim(min_e, max_e)
    axis.set_ylim(min_e, max_e)

    axis.set_xlabel('$E_{true}$ / eV')
    axis.set_ylabel('$E_{predicted}$ / eV')

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
            bins=40,
            label='$F_{x}$',
            cmap=cmaps[idx],
            norm=mpl.colors.LogNorm(),
        )

    axis.set_ylim(min_f, max_f)
    axis.set_xlim(min_f, max_f)

    axis.set_xlabel('$F_{true}$ / eV Å$^{-1}$')
    axis.set_ylabel('$F_{predicted}$ / eV Å$^{-1}$')

    return None


def _add_force_magnitude_plot(config_set, axis) -> None:
    """Add a parity plot of the force magnitudes"""

    xs, ys = [], []
    for config in config_set:
        xs += np.linalg.norm(config.forces.true, axis=1).tolist()
        ys += np.linalg.norm(config.forces.predicted, axis=1).tolist()

    min_f = min([np.min(xs), np.min(ys)])
    max_f = min([np.max(xs), np.max(ys)])

    axis.hist2d(
        xs,
        ys,
        range=[[min_f, max_f], [min_f, max_f]],
        bins=50,
        cmap=plt.get_cmap('Blues'),
        norm=mpl.colors.LogNorm(),
    )

    _add_r_sq_and_mad(axis, xs=np.array(xs), ys=np.array(ys))

    axis.set_ylim(min_f, max_f)
    axis.set_xlim(min_f, max_f)

    axis.set_xlabel('$|{\\bf{F}}|_{true}$ / eV Å$^{-1}$')
    axis.set_ylabel('$|{\\bf{F}}|_{predicted}$ / eV Å$^{-1}$')

    return None


def _add_r_sq_and_mad(axis, xs, ys):
    """Add an annotation of the correlation and MAD between the data"""

    slope, intercept, r, p, se = linregress(xs, ys)
    axis.annotate(
        f'$R^2$ = {r**2:.3f},\n' f' MAD = {np.mean(np.abs(xs - ys)):.3f} eV',
        xy=(1, 0),
        xycoords='axes fraction',
        fontsize=12,
        xytext=(-5, 5),
        textcoords='offset points',
        ha='right',
        va='bottom',
    )

    return None
