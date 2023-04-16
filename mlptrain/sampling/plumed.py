import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, List, Tuple, Optional, Union
from copy import deepcopy
from ase.calculators.plumed import Plumed
from ase.calculators.calculator import Calculator, all_changes
from ase.parallel import broadcast
from ase.parallel import world
from mlptrain.sampling._base import ASEConstraint
from mlptrain.utils import convert_ase_time
from mlptrain.log import logger


class PlumedCalculator(Plumed):
    """
    Modified ASE calculator, instead of returning biased energies and forces,
    this calculator computes unbiased energies and forces, and computes PLUMED
    energy and force biases separately.
    """
    implemented_properties = ['energy', 'forces', 'energy_bias', 'forces_bias']

    def compute_energy_and_forces(self, pos, istep) -> Tuple:
        """Compute unbiased energies and forces, and PLUMED energy and force
        biases separately"""

        unbiased_energy = self.calc.get_potential_energy(self.atoms)
        unbiased_forces = self.calc.get_forces(self.atoms)

        if world.rank == 0:
            ener_forc = self.compute_bias(pos, istep, unbiased_energy)
        else:
            ener_forc = None

        energy_bias, forces_bias = broadcast(ener_forc)
        energy = unbiased_energy
        forces = unbiased_forces

        return energy, forces, energy_bias[0], forces_bias

    def calculate(self,
                  atoms=None,
                  properties=['energy', 'forces', 'energy_bias', 'forces_bias'],
                  system_changes=all_changes
                  ) -> None:
        """Compute the properties and attach them to the results"""

        Calculator.calculate(self, atoms, properties, system_changes)

        comp = self.compute_energy_and_forces(self.atoms.get_positions(),
                                              self.istep)

        energy, forces, energy_bias, forces_bias = comp
        self.istep += 1
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['energy_bias'] = energy_bias
        self.results['forces_bias'] = forces_bias

        return None


class PlumedBias(ASEConstraint):
    """
    @DynamicAttrs
    Class for storing collective variables and parameters used in biased
    simulations
    """

    def __init__(self,
                 cvs:       Union[Sequence['_PlumedCV'], '_PlumedCV'] = None,
                 file_name: str = None):
        """
        Class for storing collective variables and parameters used in biased
        simulations, parameters are not initialised with the object and have
        to be defined seperately using PlumedBias methods.

        Can be initialised from a complete PLUMED input file as well.

        -----------------------------------------------------------------------
        Arguments:

            cvs: Sequence of PLUMED collective variables

            file_name: (str) Complete PLUMED input file
        """

        self.setup:  Optional[List[str]] = None

        self.pace:        Optional[int] = None
        self.width:       Optional[Union[Sequence[float], float]] = None
        self.height:      Optional[float] = None
        self.biasfactor:  Optional[float] = None

        self.metad_cvs:          Optional[List['_PlumedCV']] = None
        # self.plumed_calculator:  Optional['PlumedCalculator'] = None

        for param_name in ['min', 'max', 'bin', 'wstride', 'wfile', 'rfile']:
            setattr(self, f'metad_grid_{param_name}', None)

        if file_name is not None:
            self.cvs = None
            self._from_file(file_name)

        elif cvs is not None:
            cvs = self._check_cvs_format(cvs)
            self.cvs = cvs

        else:
            raise TypeError('PLUMED bias instantiation requires '
                            'a list of collective variables (CVs) '
                            'or a file containing PLUMED-type input')

    # def __deepcopy__(self, memo=None):
    #     return self

    @property
    def from_file(self) -> bool:
        """Whether the bias is initialised using PLUMED input file"""
        return self.setup is not None

    @property
    def n_cvs(self) -> int:
        """Number of collective variables attached to the bias"""
        return len(self.cvs)

    @property
    def n_metad_cvs(self) -> int:
        """Number of collective variables attached to the bias that will be
        used in metadynamics"""
        return len(self.metad_cvs)

    @property
    def cv_sequence(self) -> str:
        """String containing names of all collective variables separated
        by commas"""

        cv_names = (cv.name for cv in self.cvs)
        return ','.join(cv_names)

    @property
    def metad_cv_sequence(self) -> str:
        """String containing names of collective variables used in metadynamics
        separated by commas"""
        metad_cv_names = (cv.name for cv in self.metad_cvs)
        return ','.join(metad_cv_names)

    @property
    def metadynamics(self) -> bool:
        """True if any parameters required for metadynamics are set"""

        _metad_parameters = ['pace', 'width', 'height']

        return any(getattr(self, p) is not None for p in _metad_parameters)

    @property
    def width_sequence(self) -> str:
        """String containing width values separated by commas"""

        if self.width is None:
            raise TypeError('Width is not initialised')

        else:
            return ','.join(str(width) for width in self.width)

    @property
    def metad_grid_setup(self) -> str:
        """String specifying metadynamics grid parameters in PLUMED input
        file"""

        _metad_grid_setup = ''
        _grid_params = ['min', 'max', 'bin', 'wstride', 'wfile', 'rfile']

        for param_name in _grid_params:
            param = getattr(self, f'metad_grid_{param_name}')

            if param is not None:

                if isinstance(param, list) or isinstance(param, tuple):
                    param_str = ','.join(str(p) for p in param)

                else:
                    param_str = str(param)

                _metad_grid_setup += f'GRID_{param_name.upper()}={param_str} '

        return _metad_grid_setup

    @property
    def biasfactor_setup(self) -> str:
        """String specifying biasfactor in PLUMED input file"""

        if self.biasfactor is not None:
            return f'BIASFACTOR={self.biasfactor} '

        else:
            return ''

    def _set_metad_params(self,
                          pace:          int,
                          width:         Union[Sequence[float], float],
                          height:        float,
                          biasfactor:    Optional[float] = None,
                          cvs:           Optional = None,
                          grid_min:      Union[Sequence[float], float] = None,
                          grid_max:      Union[Sequence[float], float] = None,
                          grid_bin:      Union[Sequence[float], float] = None,
                          grid_wstride:  Optional[int] = None,
                          grid_wfile:    Optional[str] = None,
                          grid_rfile:    Optional[str] = None
                          ) -> None:
        """
        Define parameters used in (well-tempered) metadynamics.

        -----------------------------------------------------------------------
        Arguments:

            pace: (int) τ_G/dt, interval at which a new gaussian is placed

            width: (float) σ, standard deviation (parameter describing the
                           width) of the placed gaussian

            height: (float) ω, initial height of placed gaussians (in eV)

            biasfactor: (float) γ, describes how quickly gaussians shrink,
                                larger values make gaussians to be placed
                                less sensitive to the bias potential

            cvs: (mlptrain._PlumedCV) Sequence of PLUMED collective variables
                                      which will be biased. If this variable
                                      is not set all CVs attached to self
                                      will be biased

            grid_min: (float) Lower bound of the grid

            grid_max: (float) Upper bound of the grid

            grid_bin: (float) Number of bins to use for each collective
                              variable, if not specified PLUMED automatically
                              sets the width of the bin to 1/5 of the
                              width (σ) value

            grid_wstride: (float) Number of steps specifying the period at
                                  which the grid is updated

            grid_wfile: (str) Name of the file to write the grid to

            grid_rfile: (str) Name of the file to read the grid from
        """

        self._set_metad_cvs(cvs)

        if not isinstance(pace, int) or pace <= 0:
            raise ValueError('Pace (τ_G/dt) must be a positive integer')

        else:
            self.pace = pace

        if isinstance(width, list) or isinstance(width, tuple):

            if len(width) == 0:
                raise TypeError('The provided width sequence is empty')

            elif any(single_width <= 0 for single_width in width):
                raise ValueError('All gaussian widths (σ) must be positive')

            else:
                self.width = width

        else:
            if width <= 0:
                raise ValueError('Gaussian width (σ) must be positive')

            else:
                self.width = [width]

        if len(self.width) != self.n_metad_cvs:
            raise ValueError('The number of supplied widths (σ) does not '
                             'match the number of collective variables')

        if height <= 0:
            raise ValueError('Gaussian height (ω) must be positive')

        else:
            self.height = height

        if biasfactor is not None and biasfactor < 1:
            raise ValueError('Biasfactor (γ) must be larger than one')

        else:
            self.biasfactor = biasfactor

        self._set_metad_grid_params(grid_min, grid_max, grid_bin, grid_wstride,
                                    grid_wfile, grid_rfile)

        return None

    def _set_metad_cvs(self,
                       cvs: Union[Sequence['_PlumedCV'], '_PlumedCV'] = None
                       ) -> None:
        """
        Attach PLUMED collective variables to PlumedBias which will be used in
        metadynamics.

        -----------------------------------------------------------------------
        Arguments:

            cvs: (mlptrain._PlumedCV) Sequence of PLUMED collective variables
                                      which will be biased. If this variable
                                      is not set all CVs attached to self
                                      will be biased

        """

        if cvs is not None:
            cvs = self._check_cvs_format(cvs)

            for cv in cvs:
                if cv not in self.cvs:
                    raise ValueError('Supplied CVs must be a subset of CVs '
                                     'already attached to the PlumedBias')

            self.metad_cvs = cvs

        elif self.metad_cvs is not None:
            pass

        else:
            self.metad_cvs = self.cvs

        return None

    def _set_metad_grid_params(self,
                               grid_min:  Union[Sequence[float], float] = None,
                               grid_max:  Union[Sequence[float], float] = None,
                               grid_bin:  Union[Sequence[float], float] = None,
                               grid_wstride:  Optional[int] = None,
                               grid_wfile:    Optional[str] = None,
                               grid_rfile:    Optional[str] = None
                               ) -> None:
        """
        Define grid parameters used in (well-tempered) metadynamics. Grid
        bounds (min and max) must cover the whole configuration space that the
        system will explore during the simulation, otherwise PLUMED will raise
        an error.

        ------------------------------------------------------------------------
        Arguments:

            grid_min: (float) Lower bound of the grid

            grid_max: (float) Upper bound of the grid

            grid_bin: (float) Number of bins to use for each collective
                              variable, if not specified PLUMED automatically
                              sets the width of the bin to 1/5 of the
                              width (σ) value

            grid_wstride: (float) Number of steps specifying the period at
                                  which the grid is updated

            grid_wfile: (str) Name of the file to write the grid to

            grid_rfile: (str) Name of the file to read the grid from
        """

        _sequences = {'grid_min': grid_min,
                      'grid_max': grid_max,
                      'grid_bin': grid_bin}

        if grid_bin is None:
            _sequences.pop('grid_bin')

        for param_name, params in _sequences.items():

            if isinstance(params, list) or isinstance(params, tuple):

                if len(params) == 0:
                    raise ValueError('The supplied parameter sequence is empty')

                elif len(params) != self.n_metad_cvs:
                    raise ValueError('The length of the parameter sequence '
                                     'does not match the number of CVs used '
                                     'in metadynamics')

                else:
                    setattr(self, f'metad_{param_name}', params)

            elif params is not None and self.n_metad_cvs == 1:
                setattr(self, f'metad_{param_name}', [params])

        _single_params = {'grid_wstride': grid_wstride,
                          'grid_wfile': grid_wfile,
                          'grid_rfile': grid_rfile}

        for param_name, param in _single_params.items():
            if param is not None:
                setattr(self, f'metad_{param_name}', param)

        return None

    def _from_file(self, file_name) -> None:
        """Method to extract PLUMED setup from a file"""

        self.setup = []

        with open(file_name, 'r') as f:
            for line in f:
                if line[0] == '#' or line == '\n':
                    continue
                else:
                    line = line.strip()
                    self.setup.append(line)

        return None

    def initialise_for_metad_al(self,
                                width:      Union[Sequence[float], float],
                                pace:       int = 20,
                                height:     Optional[float] = None,
                                biasfactor: Optional[float] = None,
                                cvs:        Optional = None,
                                grid_min: Union[Sequence[float], float] = None,
                                grid_max: Union[Sequence[float], float] = None,
                                grid_bin: Union[Sequence[float], float] = None
                                ) -> None:
        """
        Initialise PlumedBias for metadynamics active learning by setting the
        required parameters.

        ------------------------------------------------------------------------
        Arguments:

            width: (float) σ, standard deviation (parameter describing the
                           width) of the placed gaussian

            pace: (int) τ_G/dt, interval at which a new gaussian is placed

            height: (float) ω, initial height of placed gaussians (in eV).
                            If not supplied will be set to 5*k_B*T, where
                            T is the temperature at which metadynamics active
                            learning is performed

            biasfactor: (float) γ, describes how quickly gaussians shrink,
                                larger values make gaussians to be placed
                                less sensitive to the bias potential

            cvs: (mlptrain._PlumedCV) Sequence of PLUMED collective variables
                                      which will be biased. If this variable
                                      is not set all CVs attached to self
                                      will be biased

            grid_min: (float) Lower bound of the grid

            grid_max: (float) Upper bound of the grid

            grid_bin: (float) Number of bins to use for each collective
                              variable, if not specified PLUMED automatically
                              sets bin width to 1/5 of the gaussian width (σ)
                              value
        """

        # Setting height to dummy height (otherwise _set_metad_params() method
        # complains), the true value is set in the al_train() method
        if height is None:
            dummy_height = 1E9
            height = dummy_height

        self._set_metad_params(pace, width, height, biasfactor, cvs, grid_min,
                               grid_max, grid_bin)

        return None

    def strip(self) -> None:
        """Change the bias such that it would only contain the definitions of
        collective variables and their associated upper and lower walls"""

        if self.from_file:
            self._strip_setup()

        else:
            # Setting all attributes to None, except cvs (which might have
            # walls attached)
            _attributes = deepcopy(self.__dict__)
            _attributes.pop('cvs')

            for param_name in _attributes.keys():
                setattr(self, param_name, None)

        return None

    def _strip_setup(self) -> None:
        """If the bias is initialised using a PLUMED input file, this method
        removes all lines from the setup except the ones defining lower and
        upper walls, and the collective variables to which the walls are
        attached"""

        if self.setup is None:
            raise TypeError('Setup of the bias is not initialised, if you '
                            'want to strip the setup make sure to use a bias '
                            'which was initialised using a PLUMED input file')

        _stripped_setup, _args = [], []
        for line in reversed(self.setup):

            if line.startswith('UPPER_WALLS') or line.startswith('LOWER_WALLS'):
                _stripped_setup.append(line)
                _args.extend(self._find_args(line))

            if any(line.startswith(arg) for arg in _args):
                _stripped_setup.append(line)
                _args.extend(self._find_args(line))

        # Reverse back to normal order
        _stripped_setup = _stripped_setup[::-1]

        # Remove duplicate lines
        _stripped_setup = list(dict.fromkeys(_stripped_setup))

        self.setup = _stripped_setup

        return None

    @staticmethod
    def _find_args(line) -> List:
        """Find and return inputs to ARG parameter in a given line of a PLUMED
        setup"""

        _args = []
        _line = line.split(' ')

        for element in _line:
            element = element.split('=')

            if element[0] == 'ARG':
                _args = element[-1].split(',')

        return _args

    @staticmethod
    def _check_cvs_format(cvs) -> List['_PlumedCV']:
        """Check if the supplied collective variables are in the correct
        format"""

        # e.g. cvs == [cv1, cv2]; (cv1, cv2)
        if isinstance(cvs, list) or isinstance(cvs, tuple):

            if len(cvs) == 0:
                raise TypeError('The provided collective variable '
                                'sequence is empty')

            elif all(issubclass(cv.__class__, _PlumedCV) for cv in cvs):
                pass

            else:
                raise TypeError('Supplied CVs are in incorrect format')

        # e.g. cvs == cv1
        elif issubclass(cvs.__class__, _PlumedCV):
            cvs = [cvs]

        else:
            raise TypeError('Supplied CVs are in incorrect format')

        return cvs

    def adjust_potential_energy(self, atoms) -> float:
        """Adjust the energy of the system by adding PLUMED bias"""

        # energy_bias = self.plumed_calculator.get_property(name='energy_bias',
        #                                                   atoms=atoms)
        energy_bias = atoms.calc.get_property('energy_bias', atoms)
        return energy_bias

    def adjust_forces(self, atoms, forces) -> None:
        """Adjust the forces of the system by adding PLUMED forces"""

        # forces_bias = self.plumed_calculator.get_property(name='forces_bias',
        #                                                   atoms=atoms)
        forces_bias = atoms.calc.get_property('forces_bias', atoms)
        forces += forces_bias
        return None

    def adjust_positions(self, atoms, newpositions) -> None:
        """Method required for ASE but not used in mlp-train"""
        return None


class _PlumedCV:
    """Parent class containing methods for initialising PLUMED collective
    variables"""

    def __init__(self,
                 name:         str = None,
                 atom_groups:  Sequence = None,
                 file_name:    str = None,
                 component:    Optional[str] = None):
        """
        This class contains methods to initialise PLUMED collective variables
        (CVs) and only acts as a parent class which should not be used to
        initialise CVs directly.

        CV can be initialised using groups of atoms in the same way as
        mlptrain.AverageDistance and mlptrain.DifferenceDistance are
        initialised, but this method only supports distances, angles,
        and torsions as degrees of freedom (DOFs).

        Another method to initialise a CV is to supply a PLUMED input file
        which only contains input used in the definition of the CV, and the
        CV itself.

        -----------------------------------------------------------------------
        Arguments:

            name: (str) Name of the collective variable (only for generating
                        CVs from atom_groups, the name when generating CVs from
                        a file is taken directly from the file)

            atom_groups: (Sequence[Sequence[int]]) List of atom index sequences
                          which are used to generate DOFs,
                          e.g. [(0, 1), (2, 3)];
                               [0, 1, 2]

            file_name: (str) Name of the PLUMED file used to generate a CV
                             from that file

            component: (str) Name of a component of the last CV in the supplied
                             PLUMED input file to use as a collective variable,
                             e.g. 'spath' for PATH collective variable.
        """

        self.setup = []
        self.name = self.units = self.dof_names = self.dof_units = None
        self.lower_wall = self.upper_wall = None
        self.files = None

        if file_name is not None:
            self._from_file(file_name, component)

        elif atom_groups is not None:
            self._from_atom_groups(name, atom_groups)

        else:
            raise TypeError('Collective variable instantiation requires '
                            'groups of atom indices (DOFs) '
                            'or a file containing PLUMED-type input')

    @property
    def dof_sequence(self) -> str:
        """String containing names of DOFs separated by commas"""

        return ','.join(self.dof_names)

    def attach_lower_wall(self,
                          location:  Union[float, str],
                          kappa:     float,
                          exp:       float = 2
                          ) -> None:
        """
        Attaches lower wall bias to the collective variable.

        -----------------------------------------------------------------------
        Arguments:

            location: (float | str) Value of the CV where the wall will be
                                    located

            kappa: (float) The force constant of the wall in eV/Å^(-exp) units

            exp: (float) The power of the wall
        """

        if self.lower_wall is not None:
            raise TypeError(f'Lower wall for {self.name} CV has already '
                            'been set')

        self.lower_wall = {'location': location, 'kappa': kappa, 'exp': exp}
        self.setup.extend(['LOWER_WALLS '
                           f'ARG={self.name} '
                           f'AT={location} '
                           f'KAPPA={kappa} '
                           f'EXP={exp}'])

        return None

    def attach_upper_wall(self,
                          location:  Union[float, str],
                          kappa:     float,
                          exp:       float = 2
                          ) -> None:
        """
        Attaches upper wall bias to the collective variable.

        -----------------------------------------------------------------------
        Arguments:

            location: (float | str) Value of the CV where the wall will be
                                    located

            kappa: (float) The force constant of the wall in eV/Å^(-exp) units

            exp: (float) The power of the wall
        """

        if self.upper_wall is not None:
            raise TypeError(f'Upper wall for {self.name} CV has already '
                            'been set')

        self.upper_wall = {'location': location, 'kappa': kappa, 'exp': exp}
        self.setup.extend(['UPPER_WALLS '
                           f'ARG={self.name} '
                           f'AT={location} '
                           f'KAPPA={kappa} '
                           f'EXP={exp}'])

        return None

    def _from_file(self, file_name, component) -> None:
        """Generate DOFs and a CV from a file"""

        with open(file_name, 'r') as f:
            for line in f:
                if line[0] == '#' or line == '\n':
                    continue
                else:
                    line = line.strip()
                    self.setup.extend([line])

        _names = [line.split(':')[0] for line in self.setup]

        if component is not None:
            self.name = f'{_names.pop()}.{component}'

        else:
            self.name = _names.pop()

        self._check_name()

        filenames = self._find_files()
        if len(filenames) > 0:
            self._attach_files(filenames)

        return None

    def _find_files(self) -> List:
        """Finds files in the CV initialisation"""

        filenames = []
        for line in self.setup:
            line = line.split()

            for element in line:
                element = element.split('=')
                name = element[-1]

                if name.endswith('.dat') or name.endswith('.pdb'):
                    filenames.append(name)

        return filenames

    def _attach_files(self, filenames) -> None:
        """Attaches files found in the CV initialisation to the CV object"""

        self.files = []

        for filename in filenames:
            if not os.path.exists(filename):
                raise FileNotFoundError(f'File {filename}, which is '
                                        f'required for defining the CV '
                                        f'{self.name} was not found in the '
                                        'current directory')

            with open(filename, 'r') as f:
                data = f.read()

            self.files.append((filename, data))

        return None

    def write_files(self) -> None:
        """Writes files attached to the CV to the current directory"""

        if self.files is None:
            raise TypeError(f'CV {self.name} has no attached files')

        for filename, data in self.files:
            with open(filename, 'w') as f:
                f.write(data)

        return None

    def _from_atom_groups(self, name, atom_groups) -> None:
        """Generate DOFs from atom_groups"""

        self.name = name
        self._check_name()
        self.dof_names, self.dof_units = [], []

        if isinstance(atom_groups, list) or isinstance(atom_groups, tuple):

            if len(atom_groups) == 0:
                raise TypeError('Atom groups cannot be an empty list or an '
                                'empty tuple')

            # e.g. atom_groups == [(1, 2), (3, 4)]; ([0, 1])
            elif all(isinstance(atom_group, list)
                     or isinstance(atom_group, tuple)
                     for atom_group in atom_groups):

                for idx, atom_group in enumerate(atom_groups):
                    self._atom_group_to_dof(idx, atom_group)

            # e.g. atom_groups = [0, 1]
            elif all(isinstance(idx, int) for idx in atom_groups):

                self._atom_group_to_dof(0, atom_groups)

            else:
                raise TypeError('Elements of atom_groups must all be '
                                'sequences or all be integers')

        else:
            raise TypeError('Atom groups are in incorrect format')

        return None

    def _check_name(self) -> None:
        """Checks if the supplied name is valid"""

        if ' ' in self.name:
            raise ValueError('Spaces in CV names are not allowed')

        _illegal_substrings = ['fes', 'colvar', 'HILLS']
        if any(substr in self.name for substr in _illegal_substrings):
            raise ValueError('Please do not use "fes", "colvar", "HILLS" in '
                             'your CV names')

        return None

    def _atom_group_to_dof(self, idx, atom_group) -> None:
        """Check the atom group and generate a DOF"""

        # PLUMED atom enumeration starts from 1
        atom_list = [f'{i + 1}' for i in atom_group]
        atoms = ','.join(atom_list)

        if len(atom_list) < 2:
            raise ValueError('Atom group must contain at least two atoms')

        if len(atom_list) == 2:
            dof_name = f'{self.name}_dist{idx + 1}'
            self.dof_names.append(dof_name)
            self.dof_units.append('Å')
            self.setup.extend([f'{dof_name}: '
                               f'DISTANCE ATOMS={atoms}'])

        if len(atom_list) == 3:
            dof_name = f'{self.name}_ang{idx + 1}'
            self.dof_names.append(dof_name)
            self.dof_units.append('rad')
            self.setup.extend([f'{dof_name}: '
                               f'ANGLE ATOMS={atoms}'])

        if len(atom_list) == 4:
            dof_name = f'{self.name}_tor{idx + 1}'
            self.dof_names.append(dof_name)
            self.dof_units.append('rad')
            self.setup.extend([f'{dof_name}: '
                               f'TORSION ATOMS={atoms}'])

        if len(atom_list) > 4:
            raise NotImplementedError('Instatiation using atom groups '
                                      'is only implemented for groups '
                                      'not larger than four')

        return None

    def _set_units(self, units=None) -> None:
        """Set units of the collective variable as a string"""

        if self.dof_units is not None:

            if len(set(self.dof_units)) == 1:
                self.units = set(self.dof_units).pop()

            else:
                logger.warning('DOFs in a defined CV have different units, '
                               'setting units of this CV to None')

        else:
            self.units = units

        return None


class PlumedAverageCV(_PlumedCV):
    """Class used to initialise a PLUMED collective variable as an average
    between multiple degrees of freedom"""

    def __init__(self,
                 name:         str,
                 atom_groups:  Sequence = None):
        """
        PLUMED collective variable as an average between multiple degrees of
        freedom (distances, angles, torsions),

        e.g. [(0, 1), (2, 3), (4, 5)] gives ζ = 1/3 * (r_01 + r_23 + r_45)

        -----------------------------------------------------------------------
        Arguments:

            name: (str) Name of the collective variable

            atom_groups: (Sequence[Sequence[int]]) List of atom index sequences
                                                which are used to generate DOFs
        """

        super().__init__(name=name,
                         atom_groups=atom_groups)

        self._set_units()

        dof_sum = '+'.join(self.dof_names)
        func = f'{1 / len(self.dof_names)}*({dof_sum})'

        self.setup.extend([f'{self.name}: '
                           f'CUSTOM ARG={self.dof_sequence} '
                           f'VAR={self.dof_sequence} '
                           f'FUNC={func} '
                           f'PERIODIC=NO'])


class PlumedDifferenceCV(_PlumedCV):
    """Class used to initialise a PLUMED collective variable as a difference
    between two degrees of freedom"""

    def __init__(self,
                 name:         str,
                 atom_groups:  Sequence = None):
        """
        PLUMED collective variable as a difference between two degrees of
        freedom (distances, angles, torsions),

        e.g. [(0, 1), (1, 2)] gives ζ = r_12 - r_01

        -----------------------------------------------------------------------
        Arguments:

            name: (str) Name of the collective variable

            atom_groups: (Sequence[Sequence[int]]) List of atom index sequences
                                                which are used to generate DOFs
        """

        super().__init__(name=name,
                         atom_groups=atom_groups)

        self._set_units()

        if len(self.dof_names) != 2:
            raise ValueError('DifferenceCV must comprise exactly two '
                             'groups of atoms')

        func = f'{self.dof_names[-1]}-{self.dof_names[0]}'

        self.setup.extend([f'{self.name}: '
                           f'CUSTOM ARG={self.dof_sequence} '
                           f'VAR={self.dof_sequence} '
                           f'FUNC={func} '
                           f'PERIODIC=NO'])


class PlumedCustomCV(_PlumedCV):
    """Class used to initialise a PLUMED collective variable from a file"""

    def __init__(self,
                 file_name:  str,
                 component:  Optional[str] = None,
                 units:      Optional[str] = None):
        """
        PLUMED collective variable from a file. The file must be written in the
        style of a PLUMED input file, but only contain input used in the
        definition of the CV, and the CV itself. There are no constraints
        on the CVs that can be defined.

        e.g. dof0: COM ATOMS=1-5
             dof1: CENTER ATOMS=3-7
             cv0: CUSTOM ARG=dof0,dof1 FUNC=x^2*y^2 PERIODIC=NO

        -----------------------------------------------------------------------
        Arguments:

            file_name: (str) Name of the PLUMED file used to generate a CV
                             from that file

            component: (str) Name of a component of the last CV in the supplied
                             PLUMED input file to use as a collective variable

            units: (str) Units of the collective variable, used in plots
        """
        super().__init__(file_name=file_name,
                         component=component)

        self.units = units


def plot_cv_versus_time(filename:    str,
                        style:       str = 'trajectory',
                        time_units:  str = 'ps',
                        cv_units:    Optional[str] = None,
                        cv_limits:   Optional[Sequence[float]] = None,
                        label:       Optional[str] = None,
                        ) -> None:
    """
    Plots a collective variable as a function of time from a given colvar file.
    Only plots the first collective variable in the colvar file.

    ---------------------------------------------------------------------------
    Arguments:

        filename: (str) Name of the colvar file used for plotting

        style: (str) Style to use for plotting, e.g. 'scatter', 'trajectory'

        time_units: (str) Units of time

        cv_units: Units of the CV to be plotted

        cv_limits: Min and max limits of the CV in the plot

        label: (str) Label attached to the name of the plot, useful when
                     multiple plots of the same CVs are generated in the same
                     directory
    """

    with open(filename, 'r') as f:
        header = f.readlines()[0]

    cv_name = header.split()[3]  # (#! FIELDS time cv_name ...)
    ase_time_array = np.loadtxt(filename, usecols=0)
    cv_array = np.loadtxt(filename, usecols=1)

    time_array = convert_ase_time(ase_time_array, time_units)

    fig, ax = plt.subplots()

    if style.lower() == 'scatter':
        ax.scatter(time_array, cv_array)

    if style.lower() == 'trajectory':
        ax.plot(time_array, cv_array)

    ax.set_xlabel(f'Time / {time_units}')

    if cv_units is not None:
        ax.set_ylabel(f'{cv_name} / {cv_units}')

    else:
        ax.set_ylabel(cv_name)

    if cv_limits is not None:
        ax.set(ylim=tuple(cv_limits))

    fig.tight_layout()

    if label is not None:
        fig.savefig(f'{cv_name}_{label}.pdf')

    else:
        fig.savefig(f'{cv_name}.pdf')

    plt.close(fig)

    return None


def plot_cv1_and_cv2(filenames:   Sequence[str],
                     style:       str = 'scatter',
                     cvs_units:   Optional[Sequence[str]] = None,
                     cvs_limits:  Optional[Sequence[Sequence[float]]] = None,
                     label:       Optional[str] = None
                     ) -> None:
    """
    Plots the trajectory of the system by tracking two collective variables
    using two colvar files. The function only works for two collective
    variables.

    ---------------------------------------------------------------------------
    Arguments:

        filenames: Names of two colvar files used for plotting

        style: (str) Style to use for plotting, e.g. 'scatter', 'trajectory',
                     'histogram'

        cvs_units: Units of the CVs to be plotted

        cvs_limits: Min and max limits of the CVs in the plot

        label: (str) Label attached to the name of the plot, useful when
                     multiple plots of the same CVs are generated in the same
                     directory
    """

    cvs_names, cvs_arrays = [], []

    for filename in filenames:

        with open(filename, 'r') as f:
            header = f.readlines()[0]

        cvs_names.append(header.split()[3])  # (#! FIELDS time cv_name ...)
        cvs_arrays.append(np.loadtxt(filename, usecols=1))

    fig, ax = plt.subplots()

    if style.lower() == 'scatter':
        ax.scatter(cvs_arrays[0], cvs_arrays[1])

    if style.lower() == 'trajectory':
        ax.plot(cvs_arrays[0], cvs_arrays[1])

    if style.lower() == 'histogram':
        hist = ax.hist2d(cvs_arrays[0], cvs_arrays[1], bins=50, cmin=1)

        cbar = fig.colorbar(hist[-1], ax=ax)
        cbar.set_label(label='Count')

    if cvs_units is not None:
        ax.set_xlabel(f'{cvs_names[0]} / {cvs_units[0]}')
        ax.set_ylabel(f'{cvs_names[1]} / {cvs_units[1]}')

    else:
        ax.set_xlabel(cvs_names[0])
        ax.set_ylabel(cvs_names[1])

    if cvs_limits is not None:
        ax.set(xlim=tuple(cvs_limits[0]), ylim=tuple(cvs_limits[1]))

    fig.tight_layout()

    if label is not None:
        fig.savefig(f'{cvs_names[0]}_{cvs_names[1]}_{label}.pdf')

    else:
        fig.savefig(f'{cvs_names[0]}_{cvs_names[1]}.pdf')

    plt.close(fig)

    return None
