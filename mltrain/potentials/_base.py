import numpy as np
import mltrain as mlt
from copy import deepcopy
from autode.atoms import Atom
from mltrain.log import logger
from mltrain.configurations.configuration import Configuration
from mltrain.training.active import train as al_train
from abc import ABC, abstractmethod
from typing import Optional


class MLPotential(ABC):

    def __init__(self,
                 name:   str,
                 system: 'mltrain.System'):
        """
        Machine learnt potential. Name defines the name of the potential
        which will be saved. Training data is populated

        -----------------------------------------------------------------------
        Arguments:
            name: Name of the potential

            system: System for which this potential is defined
        """
        self.name = str(name)
        self.system = system

        self._training_data = mlt.ConfigurationSet()
        self.atomic_energies = {}

    def train(self,
              configurations: Optional['mltrain.ConfigurationSet'] = None
              ) -> None:
        """
        Train this potential on a set of configurations

        -----------------------------------------------------------------------
        Arguments:
            configurations: Set of configurations to train on, if None then
                            will use self._training_data

        Raises:
            (RuntimeError):
        """
        if configurations is not None:
            self._training_data = configurations

        if len(self.training_data) == 0:
            raise RuntimeError(f'Failed to train {self.__class__.__name__}'
                               f'({self.name}) had no training configurations')

        if any(c.energy.true is None for c in self.training_data):
            raise RuntimeError('Cannot train on configurations, an '
                               'energy was undefined')

        if self.requires_atomic_energies and len(self.atomic_energies) == 0:
            raise RuntimeError(f'Cannot train {self.__class__.__name__}'
                               f'({self.name}) required atomic energies that '
                               f'are not set. Set e.g. mlp.atomic_energies '
                               '= {"H": -13.}')
        self._train()
        return None

    @abstractmethod
    def _train(self) -> None:
        """Train this potential on self._training_data"""

    @property
    @abstractmethod
    def ase_calculator(self) -> 'ase.calculators.calculator.Calculator':
        """Generate an ASE calculator for this potential"""

    @property
    @abstractmethod
    def requires_atomic_energies(self) -> bool:
        """Does this potential need E_0s for each atom to be specified"""

    @property
    @abstractmethod
    def requires_non_zero_box_size(self) -> bool:
        """Can this potential be run in a box with side lengths = 0"""

    def predict(self,
                *args) -> None:
        """
        Predict energies and forces using a MLP in serial

        -----------------------------------------------------------------------
        Arguments:
            args (mltrain.ConfigurationSet | mltrain.Configuration):
        """
        all_configurations = mlt.ConfigurationSet()

        for arg in args:
            if isinstance(arg, mlt.ConfigurationSet):
                all_configurations += arg

            elif isinstance(arg, mlt.Configuration):
                all_configurations.append(arg)

            else:
                raise ValueError('Cannot predict the energy and forces on '
                                 f'{type(arg)}')

        logger.info(f'Evaluating MLP energies over {len(all_configurations)} '
                    f'configurations')

        calculator = self.ase_calculator
        logger.info('Loaded calculator successfully')

        for configuration in all_configurations:
            atoms = configuration.ase_atoms
            atoms.set_calculator(calculator)

            # Evaluate predicted energies and forces
            configuration.energy.predicted = atoms.get_potential_energy()
            configuration.forces.predicted = atoms.get_forces()

        return None

    @property
    def training_data(self) -> 'mltrain.ConfigurationSet':
        """Training data which this potential was trained on

        Returns:
            (mltrain.ConfigurationSet):
        """
        return self._training_data

    @training_data.setter
    def training_data(self,
                      value: Optional['mltrain.ConfigurationSet']):
        """Set the training date for this MLP"""

        if value is None:
            self._training_data.clear()

        elif isinstance(value, mlt.ConfigurationSet):
            self._training_data = value

        else:
            raise ValueError(f'Cannot set the training data for {self.name} '
                             f'with {value}')

    @property
    def n_train(self) -> int:
        """
        Number of training configurations used to train this potential

        -----------------------------------------------------------------------
        Returns:
            (int):
        """
        return len(self._training_data)

    @property
    def n_eval(self) -> int:
        """
        Number of reference evaluations used to generate this potential

        -----------------------------------------------------------------------
        Returns:
            (int):
        """
        return sum(c.n_ref_evals for c in self._training_data)

    def _save_training_data_as_npz_and_xyz(self) -> None:
        """Save the training data"""

        for file_extension in ('npz', 'xyz'):
            self.training_data.save(filename=f'{self.name}_al.{file_extension}')

        return None

    def al_train(self,
                 method_name: str,
                 **kwargs
                 ) -> None:
        """
        Train this MLP using active learning (AL) using a defined reference
        method

        -----------------------------------------------------------------------
        Arguments:
            method_name:

            **kwargs:  Keyword arguments passed to mlt.training.active.train()
        """
        al_train(self, method_name=method_name, **kwargs)
        self._save_training_data_as_npz_and_xyz()

        return None

    def al_train_then_bias(self,
                           method_name: str,
                           coordinate: 'mltrain.sampling.ReactionCoordinate',
                           min_coordinate: Optional[float] = None,
                           max_coordinate: Optional[float] = None,
                           **kwargs
                           ) -> None:
        r"""
        Active learning that ensures sufficient sampling over a coordinate.
        Adds a single harmonic bias to the least well sampled regions of a
        histogram of coordinate values obtained in the AL e.g. for a
        reaction coordinate histogram that looks like:

                        |
                        |         /\      /---------
            Frequency   | \      /  |    /
                        |  \___/     |_/
                        |________________________
                                  coordinate

        then two harmonic biases would be added in the two minimums

        -----------------------------------------------------------------------
        Arguments:
            method_name: Name of the reference method to use

            coordinate: Coordinate over which to add the bias

            min_coordinate: Minimum value of the coordinate to consider
                            sampling over

            max_coordinate:
        """
        self.al_train(method_name=method_name, **kwargs)

        coords = coordinate(self._training_data)
        _max = np.max(coords) if max_coordinate is None else max_coordinate
        _min = np.min(coords) if min_coordinate is None else min_coordinate

        hist, bin_edges = np.histogram(coords, bins=np.linspace(_min, _max, 10))
        bin_centres = bin_edges[:-1] + np.diff(bin_edges)/2

        for idx, freq in enumerate(hist):
            if idx == 0 or idx == (len(hist) - 1):
                continue  # Cannot be a minimum on the first or last point

            if not (freq < hist[idx-1] and freq < hist[idx+1]):
                continue  # Not a minimum in the frequency

            target_coord = bin_centres[idx]

            logger.info('Have a minimum in the histogram of coordinates at '
                        f'x = {target_coord:.2f}. Adding a harmonic bias and '
                        f'running additional AL')

            kwargs['init_configs'] = self._best_bias_init_frame(target_coord,
                                                                coords)
            self.al_train(method_name=method_name,
                          bias=mlt.Bias(coordinate,
                                        kappa=10,                 # eV Å^-2
                                        reference=target_coord),
                          **kwargs)

        return None

    def _best_bias_init_frame(self,
                              value:  float,
                              values: np.ndarray
                              ) -> 'mltrain.configurations.ConfigurationSet':
        """
        Get the closest single frame as a configuration set to start a biased
        AL loop, where the closest distance from the value to any one of the
        values and the values are in the order defined by the trainign set

        Args:
            value:
            values:
        """
        best_idx = np.argmin(np.abs(value - values))

        return mlt.ConfigurationSet(self.training_data[best_idx])

    def set_atomic_energies(self,
                            method_name: str
                            ) -> None:
        """
        Set the atomic energies of all atoms in this system

        -----------------------------------------------------------------------
        Arguments:
            method_name: Name of the reference method to use
        """
        _spin_multiplicites = {'H': 2, 'C': 3, 'B': 2, 'N': 4, 'O': 3, 'F': 2,
                               'Si': 3, 'P': 4, 'S': 3, 'Cl': 2, 'I': 2}

        for symbol in self.system.unique_atomic_symbols:
            config = Configuration(atoms=[Atom(symbol)],
                                   charge=0,
                                   mult=_spin_multiplicites[symbol])

            config.single_point(method=method_name, n_cores=1)

            if config.energy.true is None:

                if symbol == 'H':
                    logger.warning('Using estimated H atom ground state E')
                    config.energy.true = -13.6056995   # -0.5 Ha

                else:
                    raise RuntimeError('Failed to calculate an energy for '
                                       f'{symbol}')

            self.atomic_energies[symbol] = config.energy.true

        return None

    def copy(self) -> 'MLPotential':
        return deepcopy(self)
