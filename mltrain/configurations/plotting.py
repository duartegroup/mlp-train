import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress


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


def parity(config_set: 'mltrain.ConfigurationSet',
           name:       str = 'paritiy') -> None:
    """
    Plot parity plots of energies, forces and temporal differences (if present)
    otherwise the residuals over the configuration index

    ---------------------------------------------------------------------------
    Arguments:
        config_set:

    Keyword Arguments:
        name:
    """
    fig, ax = plt.subplots(nrows=2, ncols=2)

    if _all_energies_are_defined(config_set):
        _add_energy_time_plot(config_set, axis=[0, 0])
        _add_energy_parity_plot(config_set, axis=ax[0, 1])

    if _all_forces_are_defined(config_set):
        _add_force_component_plot(config_set, axis=[1, 0])
        _add_force_magnitude_plot(config_set, axis=[1, 0])

    plt.savefig(f'{name}.pdf')
    return None


def _all_energies_are_defined(cfgs) -> bool:
    """Are all the energies defined in a configuration set?"""
    return any(e is None for e in cfgs.true_energies + cfgs.predicted_energies)


def _all_forces_are_defined(cfgs) -> bool:
    """Are all the forces defined in a configuration set?"""
    return cfgs.true_forces is not None and cfgs.predicted_forces is not None


def _add_energy_time_plot(config_set, axis) -> None:


    return None


def _add_energy_parity_plot(config_set, axis) -> None:
    """Plot true vs predicted energies"""
    x, y = config_set.true_energies, config_set.predicted_energies

    axis.scatter(x, y,
                 marker='o',
                 ms=10,
                 mfc='white')

    slope, intercept, r, p, se = linregress(x, y)
    axis.annotate(f'$R^2$ = {r**2}, MAD = {np.mean(np.abs(x - y)):.4} eV',
                  xy=(1, 0),
                  xycoords='axes fraction',
                  fontsize=16,
                  xytext=(-5, 5),
                  textcoords='offset points',
                  ha='right',
                  va='bottom')

    return None
