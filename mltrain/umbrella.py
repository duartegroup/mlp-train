import matplotlib.pyplot as plt
from mltrain.bias import Bias
from mltrain.configurations import ConfigurationSet
from mltrain.md import run_mlp_md
from mltrain.log import logger
from copy import deepcopy
import numpy as np
from typing import Optional
from scipy.optimize import curve_fit


def _get_rxn_coords(atoms, atom_pair_list):
    """Return the average distance between atoms in all m pairs"""

    euclidean_dists = [atoms.get_distance(i, j, mic=True)
                       for (i, j) in atom_pair_list]

    return np.mean(euclidean_dists)


class UmbrellaSampling:
    """
    Umbrella sampling class for generating pulling simulation, running
    umbrella sampling windows and running WHAM or umbrella integration.
    """

    def __init__(self,
                 kappa: float,
                 reference: Optional[float] = None,
                 **kwargs):
        """
        Umbrella sampling to predict free energy using an mlp under a harmonic
        bias: ω = κ/2 (f(r) - ref)^2

        e.g. umbrella = mlt.umbrella.Umbrella(to_average=[[5, 1]], reference=5,
                                              kappa=10)

        -----------------------------------------------------------------------
        Arguments:

            kappa: Value of the spring_const, κ, used in umbrella sampling

            reference: Value of the reference value, ξ_i, used in umbrella
                       sampling

        -------------------
        Keyword Arguments:

            {to_add, to_subtract, to_average}: (list) Indicies of the atoms
            which are combined in some way to define the reaction rxn_coord
        """

        self.kappa = kappa
        self.bias = Bias(kappa=self.kappa, reference=reference, **kwargs)

        if 'to_add' in kwargs:
            self.rxn_coord = kwargs['to_add']
            raise NotImplementedError("Addition reaction rxn_coord not yet "
                                      "implemented")

        elif 'to_subtract' in kwargs:
            self.rxn_coord = kwargs['to_subtract']
            raise NotImplementedError("Subtract reaction rxn_coord not yet "
                                      "implemented")

        elif 'to_average' in kwargs:
            self.rxn_coord = kwargs['to_average']

        else:
            raise ValueError("Bias must specify one of to_add, to_subtract "
                             "or to_average!")

        self.num_pairs = len(self.rxn_coord)
        self.refs = None

    def _get_window_frames(self, traj, num_windows, init_ref, final_ref):
        """Returns the set of frames with reference values to use in the US"""

        # Get a dictonary of reaction atom_pair_list distances for each frame
        traj_dists = {}
        for index, frame in enumerate(traj):

            avg_distance = _get_rxn_coords(frame.ase_atoms, self.rxn_coord)
            traj_dists[index] = avg_distance

        if init_ref is None:
            init_ref = list(traj_dists.values())[0]

        if final_ref is None:
            final_ref = list(traj_dists.values())[-1]

        self.refs = np.linspace(init_ref, final_ref, num_windows)

        # Get the frames to be used in the umbrella sampling windows
        frames = ConfigurationSet()
        for reference in self.refs:

            window_dists = deepcopy(traj_dists)

            for frame_key, dist_value in window_dists.items():
                window_dists[frame_key] = abs(dist_value - reference)

            traj_idx = min(window_dists.keys(), key=window_dists.get)

            frames = frames + traj[traj_idx]

        if len(frames) < num_windows:
            logger.warning("Number of configurations extracted to run "
                           "umbrella sampling < number of windows specified "
                           "as at least two configurations were identical")

        return frames

    def _run_individual_window(self, frame, mlp, temp, interval, dt, ref,
                               **kwargs):
        """Runs an individual umbrella sampling window. Adaptive sampling to
        be implemented"""

        self.bias.kappa = self.kappa
        self.bias.ref = ref

        traj = run_mlp_md(configuration=frame,
                          mlp=mlp,
                          temp=temp,
                          dt=dt,
                          interval=interval,
                          bias=self.bias,
                          **kwargs)

        return traj

    def _fit_gaussian(self, data):
        """Fit a Gaussian to a set of data"""

        def gauss(x, a, b, c):
            return a * np.exp(-(x - b)**2 / (2. * c**2))

        min_x = min(self.refs) * 0.9
        max_x = max(self.refs) * 1.1

        x_range = np.linspace(min_x, max_x, 500)

        hist, bin_edges = np.histogram(data, density=False, bins=500)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

        initial_guess = [1.0, 1.0, 1.0]
        parms, _ = curve_fit(gauss, bin_centres, hist, p0=initial_guess,
                             maxfev=10000)

        hist_fit = gauss(x_range, parms[0], parms[1], np.abs(parms[2]))

        plt.plot(bin_centres, hist, alpha=0.1)
        plt.plot(x_range, hist_fit)

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')

        plt.savefig('fitted_data.pdf', dpi=300)

        return parms

    def run_umbrella_sampling(self, traj, mlp, temp, interval, dt,
                              num_windows=10, init_ref=None, final_ref=None,
                              **kwargs):
        """Run umbrella sampling across n windows."""

        frames = self._get_window_frames(traj, num_windows,
                                         init_ref, final_ref)

        combined_traj = ConfigurationSet()
        for win_idx, frame in enumerate(frames):

            logger.info(f'Running US window {win_idx} with reference '
                        f'{self.refs[win_idx]:.2f} Å and '
                        f'kappa {self.kappa} eV / Å^2')

            window_traj = self._run_individual_window(frame,
                                                      mlp,
                                                      temp,
                                                      interval,
                                                      dt,
                                                      ref=self.refs[win_idx],
                                                      **kwargs)

            win_rxn_coords = [_get_rxn_coords(config.ase_atoms, self.rxn_coord)
                              for config in window_traj]
            self._fit_gaussian(win_rxn_coords)

            combined_traj = combined_traj + window_traj

        combined_traj.save(filename='combined_windows.xyz')

        return None

    def calculate_free_energy(self):
        """Calculates the free energy using WHAM or umbrella Integration"""

        return NotImplementedError
