import os
import numpy as np
from time import time
from multiprocessing import Pool
from typing import Optional, List, Union
from mltrain.config import Config
from mltrain.log import logger
from mltrain.configurations.plotting import parity_plot


class ConfigurationSet(list):
    """A set of configurations"""

    @property
    def true_energies(self) -> List[Optional[float]]:
        """True calculated energies"""
        return [c.energy.true for c in self]

    @property
    def true_forces(self) -> Optional[np.ndarray]:
        """
        True force tensor. shape = (N, n_atoms, 3)

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray | None)
        """
        return self._forces('true')

    @property
    def predicted_energies(self) -> List[Optional[float]]:
        """Predicted energies using a MLP"""
        return [c.energy.predicted for c in self]

    @property
    def predicted_forces(self) -> Optional[np.ndarray]:
        """
        Predicted force tensor. shape = (N, n_atoms, 3)

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray | None)
        """
        return self._forces('predicted')

    @property
    def lowest_energy(self) -> 'mltrain.Configuration':
        """
        Determine the lowest energy configuration in this set based on the
        true energies. If not evaluated then returns the first configuration

        -----------------------------------------------------------------------
        Returns:
            (mltrain.Configuration):
        """
        if len(self) == 0:
            raise ValueError('No lowest energy configuration in an empty set')

        energies = [e if e is not None else np.inf for e in self.true_energies]
        return self[np.argmin(energies)]

    def append(self,
               value: Optional['mltrain.Configuration']) -> None:
        """
        Append an item onto these set of configurations. None will not be
        appended

        -----------------------------------------------------------------------
        Arguments:
            value:
        """

        if value is None:
            return

        return super().append(value)

    def compare(self,
                *args: Union['mltrain.potentials.MLPotential', str]) -> None:
        """
        Compare methods e.g. a MLP to a ground truth reference method over
        these set of configurations. Will generate plots of total energies
        over these configurations and save a text file with ∆s

        -----------------------------------------------------------------------
        Arguments:
            *args: Strings defining the method or MLPs
        """
        if len([arg for arg in args if isinstance(arg, str)]) > 1:
            raise NotImplementedError('Compare currently only supports a '
                                      'single reference method (string).')

        name = self._comparison_name(*args)

        if os.path.exists(f'{name}.npz'):
            self.load_npz(f'{name}.npz')

        else:
            for arg in args:
                if hasattr(arg, 'predict'):
                    arg.predict(self)

                elif isinstance(arg, str):
                    self.single_point(method_name=arg)

                else:
                    raise ValueError(f'Cannot compare using {arg}')

            self.save_npz(filename=f'{name}.npz')

        parity_plot(self, name=name)
        return None

    def save(self,
             filename:  str,
             true:      bool = False,
             predicted: bool = False
             ) -> None:
        """Save these configurations to a file

        -----------------------------------------------------------------------
        Arguments:
            filename:

            true: Save 'true' energies and forces, if they exist

            predicted: Save the MLP predicted energies and forces, if they
                       exist.
        """

        if len(self) == 0:
            logger.error(f'Failed to save {filename}. Had no configurations')
            return

        if self[0].energy.true is not None and not (predicted or true):
            logger.warning('Save called without defining what energy and '
                           'forces to print. Had true energies to using those')
            true = True

        # Empty the file
        open(filename, 'w').close()

        for configuration in self:
            configuration.save(filename,
                               true=true,
                               predicted=predicted,
                               append=True)
        return None

    def save_npz(self, filename: str) -> None:
        """Save a set of parameters for this configuration"""

        np.savez(filename,
                 E_true=self.true_energies,
                 E_predicted=self.predicted_energies,
                 F_true=self.true_forces,
                 F_predicted=self.predicted_forces)

        return None

    def load_npz(self, filename: str) -> None:
        """Load energies and forces from a saved numpy file"""

        data = np.load(filename)

        for i, config in enumerate(self):
            config.energy.true = data['E_true'][i]
            config.energy.predicted = data['E_predicted'][i]

            config.forces.true = data['F_true'][i]
            config.forces.predicted = data['F_predicted'][i]

        return None

    def single_point(self,
                     method_name: str) -> None:
        """
        Evaluate energies and forces on all configuration in this set

        -----------------------------------------------------------------------
        Arguments:
            method_name:
        """
        return self._run_parallel_method(function=_single_point_eval,
                                         method_name=method_name)

    def _forces(self, kind: str) -> Optional[np.ndarray]:
        """True or predicted forces. Returns a 3D tensor"""

        all_forces = []
        for config in self:
            if getattr(config.forces, kind) is None:
                logger.error(f'{kind} forces not defined - returning None')
                return None

            all_forces.append(getattr(config.forces, kind))

        return np.array(all_forces)

    def __add__(self,
                other: Union['mltrain.Configuration',
                             'mltrain.ConfigurationSet']
                ):
        """Add another configuration or set of configurations onto this one"""

        if other.__class__.__name__ == 'Configuration':
            self.append(other)

        elif isinstance(other, ConfigurationSet):
            self.extend(other)

        else:
            raise TypeError('Can only add a Configuration or'
                            f' ConfigurationSet, not {type(other)}')

        logger.info(f'Current number of configurations is {len(self)}')
        return self

    def _run_parallel_method(self, function, **kwargs):
        """Run a set of electronic structure calculations on this set
        in parallel

        -----------------------------------------------------------------------
        Arguments
            function: A method to calculate energy and forces on a configuration
        """
        logger.info(f'Running calculations over {len(self)} configurations\n'
                    f'Using {Config.n_cores} total cores')

        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MLK_NUM_THREADS'] = '1'

        start_time = time()
        results = []

        with Pool(processes=Config.n_cores) as pool:

            for _, config in enumerate(self):
                result = pool.apply_async(func=function,
                                          args=(config,),
                                          kwds=kwargs)
                results.append(result)

            for i, result in enumerate(results):
                self[i] = result.get(timeout=None)

        logger.info(f'Calculations done in {(time() - start_time) / 60:.1f} m')
        return None

    @staticmethod
    def _comparison_name(*args):
        """Name of a comparison between different methods"""

        name = ''
        for arg in args:
            if hasattr(arg, 'predict'):
                name += arg.name

            if isinstance(arg, str):
                name += f'_{arg}'

        return name


def _single_point_eval(config, method_name, **kwargs):
    """Top-level hashable function useful for multiprocessing"""
    config.single_point(method_name, **kwargs)
    return config
