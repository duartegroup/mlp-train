import matplotlib.pyplot as plt
import mltrain
from mltrain.bias import Bias
from mltrain.configurations import ConfigurationSet
from mltrain.md import run_mlp_md
from mltrain.log import logger
import numpy as np
from typing import Optional
from scipy.optimize import curve_fit


def _get_avg_dists(atoms, atom_pair_list):
    """Return the average distance between atoms in all m pairs"""

    euclidean_dists = [atoms.get_distance(i, j, mic=True)
                       for (i, j) in atom_pair_list]

    return np.mean(euclidean_dists)


def _run_individual_window(frame, mlp, temp, interval, dt, bias, **kwargs):
    """Runs an individual umbrella sampling window. Adaptive sampling to
    be implemented"""

    traj = run_mlp_md(configuration=frame,
                      mlp=mlp,
                      temp=temp,
                      dt=dt,
                      interval=interval,
                      bias=bias,
                      **kwargs)

    return traj


class UmbrellaSampling:
    """
    Umbrella sampling class for generating pulling simulation, running
    umbrella sampling windows and running WHAM or umbrella integration.
    """

    def __init__(self,
                 kappa: float,
                 **kwargs):
        """
        Umbrella sampling to predict free energy using an mlp under a harmonic
        bias: ω = κ/2 (f(r) - ref)^2

        e.g. umbrella = mlt.umbrella.Umbrella(to_average=[[5, 1]], reference=5,
                                              kappa=10)

        -----------------------------------------------------------------------
        Arguments:

            kappa: Value of the spring_const, κ, used in umbrella sampling

        -------------------
        Keyword Arguments:

            {to_add, to_subtract, to_average}: (list) Indicies of the atoms
            which are combined in some way to define the reaction rxn_coord
        """

        self.kappa = kappa
        self.refs = None
        self.rxn_coord_type = kwargs
        self.parm_list = []

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

    @property
    def _n_pairs(self):
        """Number of atom pairs defined in the reaction coordinate"""
        return len(self.rxn_coord)

    def _get_closest_frames(self, traj, distances):
        """Find the frames whose reaction coordinate is closest to the ref"""

        frames = ConfigurationSet()
        for reference in self.refs:

            window_dists = []
            for dist in distances:
                window_dists.append(abs(dist - reference))

            frames = frames + traj[np.argmin(window_dists)]

        return frames

    def _get_window_frames(self, traj, num_windows, init_ref, final_ref):
        """Returns the set of frames with reference values to use in the US"""

        traj_dists = []
        for index, frame in enumerate(traj):

            avg_distance = _get_avg_dists(frame.ase_atoms, self.rxn_coord)
            traj_dists.append(avg_distance)

        if init_ref is None:
            init_ref = traj_dists[0]

        if final_ref is None:
            final_ref = traj_dists[-1]

        self.refs = np.linspace(init_ref, final_ref, num_windows)

        if any(self.refs) < 0:
            raise ValueError("Reference values must be positive")

        frames = self._get_closest_frames(traj, traj_dists)

        if len(frames) < num_windows:
            logger.warning("Number of configurations extracted to run "
                           "umbrella sampling < number of windows specified "
                           "as at least two configurations were identical")

        return frames

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

        plt.savefig('fitted_data.pdf')

        return parms

    def run_umbrella_sampling(self,
                              traj: 'mltrain.ConfigurationSet',
                              mlp: 'mltrain.potentials._base.MLPotential',
                              temp: float,
                              interval: int,
                              dt: float,
                              init_ref: Optional[float] = None,
                              final_ref: Optional[float] = None,
                              n_windows: int = 10,
                              **kwargs) -> None:
        """
        Run umbrella sampling across n windows, fitting Gaussians to the data

        -----------------------------------------------------------------------
        Arguments:
            traj:

            mlp: 

            temp: (float) Temperature in K to initialise velocities and to run
                  NVT MD, if temp=0 then will run NVE
            
            interval: (int) Interval between saving the geometry
            
            dt: (float) Time-step in fs
            
            init_ref: (float | None) Value of reaction coordinate in Å for
                       first window
            
            final_ref: (float | None) Value of reaction coordinate in Å for
                       first window
            
            n_windows: (int) Number of windows to run in the umbrella sampling

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units

        Returns:
            None:
        """

        frames = self._get_window_frames(traj, n_windows,
                                         init_ref, final_ref)

        combined_traj = ConfigurationSet()
        for win_idx, frame in enumerate(frames):

            bias = Bias(self.kappa, self.refs[win_idx], **self.rxn_coord_type)

            logger.info(f'Running US window {win_idx} with reference '
                        f'{self.refs[win_idx]:.2f} Å and '
                        f'κ {self.kappa} eV / Å^2')

            window_traj = _run_individual_window(frame,
                                                 mlp,
                                                 temp,
                                                 interval,
                                                 dt,
                                                 bias=bias,
                                                 **kwargs)

            win_rxn_coords = [_get_avg_dists(config.ase_atoms, self.rxn_coord)
                              for config in window_traj]

            parms = self._fit_gaussian(win_rxn_coords)
            self.parm_list.append(parms)

            combined_traj = combined_traj + window_traj

        combined_traj.save(filename='combined_windows.xyz')

        return None
