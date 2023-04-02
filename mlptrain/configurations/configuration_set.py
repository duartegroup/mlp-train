import os
import numpy as np
from time import time
from multiprocessing import Pool
from typing import Optional, List, Union
from autode.atoms import elements, Atom
from mlptrain.config import Config
from mlptrain.log import logger
from mlptrain.configurations.configuration import Configuration
from mlptrain.configurations.plotting import parity_plot
from mlptrain.box import Box


class ConfigurationSet(list):
    """A set of configurations"""

    def __init__(self,
                 *args: Union[Configuration, str]):
        """
        Construct a configuration set from Configurations, or a saved file.
        This is a set, thus no duplicates configurations are present.

        -----------------------------------------------------------------------
        Arguments:
            args: Either strings of existing files (e.g. data.npz) or
                  individual configurations.
        """
        super().__init__()

        for arg in args:

            if isinstance(arg, Configuration):
                self.append(arg)

            elif isinstance(arg, str) and arg.endswith('.npz'):
                self.load(arg)

            else:
                raise ValueError(f'Cannot create configurations from {arg}')

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
    def bias_energies(self) -> List[Optional[float]]:
        """Bias energies from ASE and PLUMED biases"""
        return [c.energy.bias for c in self]

    @property
    def inherited_bias_energies(self) -> List[Optional[float]]:
        """Inherited metadynamics bias during active learning"""
        return [c.energy.inherited_bias for c in self]

    @property
    def lowest_energy(self) -> 'mlptrain.Configuration':
        """
        Determine the lowest energy configuration in this set based on the
        true energies. If not evaluated then returns the first configuration

        -----------------------------------------------------------------------
        Returns:
            (mlptrain.Configuration):
        """
        if len(self) == 0:
            raise ValueError('No lowest energy configuration in an empty set')

        energies = [e if e is not None else np.inf for e in self.true_energies]
        return self[np.argmin(energies)]

    def lowest_biased_energy(self, inherited) -> 'mlptrain.Configuration':
        """
        Determine the lowest biased energy configuration in this set based on
        the true and bias energies. If not evaluated then returns the first
        configuration

        -----------------------------------------------------------------------
        Arguments:

            inherited: (bool) If True then inherited bias energies are used
                              instead of bias energies

        ---------------
        Returns:
            (mlptrain.Configuration):
        """
        if len(self) == 0:
            raise ValueError('No lowest biased energy configuration in an '
                             'empty set')

        true_energy = np.array([e if e is not None else np.inf
                                for e in self.true_energies])

        if inherited:
            bias_energy = np.array([e if e is not None else 0
                                    for e in self.inherited_bias_energies])

        else:
            bias_energy = np.array([e if e is not None else 0
                                    for e in self.bias_energies])

        biased_energy = true_energy + bias_energy
        return self[np.argmin(biased_energy)]

    @property
    def has_a_none_energy(self) -> bool:
        """
        Does this set of configurations have a true energy that is undefined
        (i.e. thus is set to None)?

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        return any(c.energy.true is None for c in self)

    def remove_none_energy(self) -> None:
        """
        Remove configurations in this set with no true energy
        """

        # Delete in reverse order to preserve indexing once and item is deleted
        for idx in reversed(range(len(self))):
            if self[idx].energy.true is None:
                del self[idx]

        return None

    def remove_above_e(self, energy: float) -> None:
        """
        Remove all configuration above a particular *relative* energy

        -----------------------------------------------------------------------
        Arguments:
            energy: Relative energy (eV) above which to discard configurations
        """
        min_energy = self.lowest_energy.energy.true

        for idx in reversed(range(len(self))):
            if (self[idx].energy.true - min_energy) > energy:
                del self[idx]

        return None

    def t_min(self, from_idx: int) -> float:
        """
        Determine the minimum time for a slice (portion) of these
        configurations, if a time is not specified for a frame then assume
        it was generated at 'zero' time

        -----------------------------------------------------------------------
        Arguments:
            from_idx: Index from which to consider the minimum time

        Returns:
            (float): Time in fs
        """
        if len(self) < from_idx:
            logger.warning('Insufficient data to determine minimum time '
                           f'from index {from_idx}')
            return 0.0

        return min(c.time if c.time is not None else 0.0
                   for c in self[from_idx:])

    def append(self,
               value: Optional['mlptrain.Configuration']
               ) -> None:
        """
        Append an item onto these set of configurations. None will not be
        appended

        -----------------------------------------------------------------------
        Arguments:
            value: Configuration
        """

        if value is None:
            return

        if value in self:
            logger.warning('Not appending configuration to set - already '
                           'present')
            return

        return super().append(value)

    def compare(self,
                *args: Union['mlptrain.potentials.MLPotential', str]
                ) -> None:
        """
        Compare methods e.g. a MLP to a ground truth reference method over
        these set of configurations. Will generate plots of total energies
        over these configurations and save a text file with ∆s

        -----------------------------------------------------------------------
        Arguments:
            *args: Strings defining the method or MLPs
        """
        if _num_strings_in(args) > 1:
            raise NotImplementedError('Compare currently only supports a '
                                      'single reference method (string).')

        name = self._comparison_name(*args)

        if os.path.exists(f'{name}.npz'):
            self.load(f'{name}.npz')

        else:
            for arg in args:
                if hasattr(arg, 'predict'):
                    arg.predict(self)

                elif isinstance(arg, str):
                    self.single_point(method=arg)

                else:
                    raise ValueError(f'Cannot compare using {arg}')

            self.save(filename=f'{name}.npz')

        parity_plot(self, name=name)
        return None

    def save_xyz(self,
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
            return None

        if self[0].energy.true is not None and not (predicted or true):
            logger.warning('Save called without defining what energy and '
                           'forces to print. Had true energies to using those')
            true = True

        open(filename, 'w').close()  # Empty the file

        for configuration in self:
            configuration.save_xyz(filename,
                                   true=true,
                                   predicted=predicted,
                                   append=True)
        return None

    def load_xyz(self,
                 filename: str,
                 charge:   int,
                 mult:     int,
                 box:      Optional[Box] = None
                 ) -> None:
        """
        Load configurations from a .xyz file. Will not load any energies or
        forces

        -----------------------------------------------------------------------
        Arguments:
            filename:

            charge: Total charge on the configuration

            mult: Total spin multiplicity

            box: Box or None, if the configurations are in vacuum
        """
        file_lines = open(filename, 'r', errors='ignore').readlines()
        atoms = []

        def is_xyz_line(_l):
            return len(_l.split()) in (4, 7) and _l.split()[0] in elements

        def append_configuration(_atoms):
            self.append(Configuration(_atoms, charge, mult, box))
            return None

        for idx, line in enumerate(file_lines):

            if is_xyz_line(line):
                atoms.append(Atom(*line.split()[:4]))

            elif len(atoms) > 0:
                append_configuration(atoms)
                atoms.clear()

        if len(atoms) > 0:
            append_configuration(atoms)

        return None

    def save(self, filename: str) -> None:
        """
        Save all the parameters for this configuration. Overrides any current
        data in that file

        -----------------------------------------------------------------------
        Arguments:
            filename: Filename, if extension is not .xyz it will be added
        """

        if len(self) == 0:
            logger.error('Configuration set had no components, not saving')
            return

        if filename.endswith('.xyz'):
            self.save_xyz(filename)

        elif filename.endswith('.npz'):
            self._save_npz(filename)

        else:
            logger.warning('Filename had no valid extension - adding .npz')
            self._save_npz(f'{filename}.npz')

        return None

    def load(self, filename: str) -> None:
        """
        Load energies and forces from a saved numpy compressed array.

        -----------------------------------------------------------------------
        Arguments:
            filename:

        Raises:
            (ValueError): If an unsupported file extension is present
        """

        if filename.endswith('.npz'):
            self._load_npz(filename)

        elif filename.endswith('.xyz'):
            raise ValueError('Loading .xyz files is not supported. Call '
                             'load_xyz() with defined charge & multiplicity')

        else:
            raise ValueError(f'Cannot load {filename}. Must be either a '
                             f'.xyz or .npz file')

        return None

    def single_point(self,
                     method: str) -> None:
        """
        Evaluate energies and forces on all configuration in this set

        -----------------------------------------------------------------------
        Arguments:
            method:
        """
        return self._run_parallel_method(function=_single_point_eval,
                                         method_name=method)

    @property
    def _coordinates(self) -> np.ndarray:
        """
        Coordinates of all the configurations in this set

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): Coordinates tensor (n, n_atoms, 3)
        """
        return np.array([np.asarray(c.coordinates, dtype=float) for c in self])

    @property
    def plumed_coordinates(self) -> Optional[np.ndarray]:
        """
        PLUMED collective variable values in this set

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): PLUMED collective variable tensor (n, n_cvs)
        """

        n_cvs = 0
        all_coordinates = []

        for config in self:
            all_coordinates.append(config.plumed_coordinates)

            if config.plumed_coordinates is not None:
                n_cvs = len(config.plumed_coordinates)

        if n_cvs == 0:
            logger.info(f'PLUMED coordinates not defined - returning None')
            return None

        for i, coords in enumerate(all_coordinates):
            if coords is None:
                all_coordinates[i] = np.array([np.nan for _ in range(n_cvs)])

        return np.array(all_coordinates)

    @property
    def _atomic_numbers(self) -> np.ndarray:
        """
        Atomic numbers of atoms in all the configurations in this set

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): Atomic numbers matrix (n, n_atoms)
        """

        return np.array([[atom.atomic_number for atom in c.atoms] for c in self])

    @property
    def _box_sizes(self) -> np.ndarray:
        """
        Box sizes of all the configurations in this set, if a configuration
        does not have a box then use a zero set of lattice lengths.

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): Box sizes matrix (n, 3)
        """
        return np.array([c.box.size if c.box is not None else np.zeros(3)
                         for c in self])

    @property
    def _charges(self) -> np.ndarray:
        """Total charges of all configurations in this set"""
        return np.array([c.charge for c in self])

    @property
    def _multiplicities(self) -> np.ndarray:
        """Total spin multiplicities of all configurations in this set"""
        return np.array([c.mult for c in self])

    def _forces(self, kind: str) -> Optional[np.ndarray]:
        """True or predicted forces. Returns a 3D tensor"""

        all_forces = []
        for config in self:
            if getattr(config.forces, kind) is None:
                logger.error(f'{kind} forces not defined - returning None')
                return None

            all_forces.append(getattr(config.forces, kind))

        return np.array(all_forces)

    def _save_npz(self, filename: str) -> None:
        """Save a compressed numpy array of all the data in this set"""

        np.savez(filename,
                 R=self._coordinates,
                 R_plumed=self.plumed_coordinates,
                 E_true=self.true_energies,
                 E_predicted=self.predicted_energies,
                 E_bias=self.bias_energies,
                 E_inherited_bias=self.inherited_bias_energies,
                 F_true=self.true_forces,
                 F_predicted=self.predicted_forces,
                 Z=self._atomic_numbers,
                 L=self._box_sizes,
                 C=self._charges,
                 M=self._multiplicities,
                 allow_pickle=True)

        return None

    def _load_npz(self, filename: str) -> None:
        """Load a compressed numpy array of all the data in this set"""

        data = np.load(filename, allow_pickle=True)

        for i, coords in enumerate(data['R']):

            box = Box(size=data['L'][i])

            config = Configuration(atoms=_atoms_from_z_r(data['Z'][i], coords),
                                   charge=int(data['C'][i]),
                                   mult=int(data['M'][i]),
                                   box=None if box.has_zero_volume else box)

            if data['R_plumed'].ndim > 0:
                config.plumed_coordinates = data['R_plumed'][i]

            if data['E_true'].ndim > 0:
                config.energy.true = data['E_true'][i]

            if data['E_predicted'].ndim > 0:
                config.energy.predicted = data['E_predicted'][i]

            if data['E_bias'].ndim > 0:
                config.energy.bias = data['E_bias'][i]

            if data['E_inherited_bias'].ndim > 0:
                config.energy.inherited_bias = data['E_inherited_bias'][i]

            if data['F_true'].ndim > 0:
                config.forces.true = data['F_true'][i]

            if data['F_predicted'].ndim > 0:
                config.forces.predicted = data['F_predicted'][i]

            self.append(config)

        return None

    def __add__(self,
                other: Union['mlptrain.Configuration',
                             'mlptrain.ConfigurationSet']):
        """Add another configuration or set of configurations onto this one"""

        if isinstance(other, Configuration):
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
        logger.info(f'Running calculations over {len(self)} configurations')

        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MLK_NUM_THREADS'] = '1'

        start_time = time()
        results = []

        n_processes = min(len(self), Config.n_cores)
        n_cores_pp = max(Config.n_cores // len(self), 1)
        kwargs['n_cores'] = n_cores_pp
        logger.info(f'Running {n_processes} processes; {n_cores_pp} cores each')

        with Pool(processes=n_processes) as pool:

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


def _atoms_from_z_r(atomic_numbers: np.ndarray,
                    coordinates:    np.ndarray) -> List[Atom]:
    """From a set of atomic numbers and coordinates create a set of atoms"""

    atoms = []

    for atomic_number, coord in zip(atomic_numbers, coordinates):
        atoms.append(Atom(elements[atomic_number - 1], *coord))

    return atoms


def _num_strings_in(_list):
    """Number of strings in a list"""
    return len([item for item in _list if isinstance(item, str)])
