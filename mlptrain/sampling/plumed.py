import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Union
from mlptrain.utils import convert_ase_time
from mlptrain.log import logger


class PlumedBias:
    """Class for storing collective variables and parameters used in biased
    simulations"""

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

        self.setup = None
        self.pace = self.width = self.height = self.biasfactor = None

        if file_name is not None:
            self._from_file(file_name)

        elif cvs is not None:

            # e.g. cvs == [cv1, cv2]; (cv1, cv2)
            if isinstance(cvs, list) or isinstance(cvs, tuple):

                if len(cvs) == 0:
                    raise TypeError('The provided collective variable '
                                    'sequence is empty')

                elif all(issubclass(cv.__class__, _PlumedCV) for cv in cvs):
                    self.cvs = cvs

                else:
                    raise TypeError('Supplied CVs are in incorrect format')

            # e.g. cvs == cv1
            elif issubclass(cvs.__class__, _PlumedCV):
                self.cvs = [cvs]

            else:
                raise TypeError('Supplied CVs are in incorrect format')

        else:
            raise TypeError('PLUMED bias instantiation requires '
                            'a list of collective variables (CVs) '
                            'or a file containing PLUMED-type input')

    @property
    def cv_sequence(self) -> str:
        """String containing names of collective variables separated
        by commas"""
        cv_names = (cv.name for cv in self.cvs)
        return ','.join(cv_names)

    @property
    def width_sequence(self) -> str:
        """String containing width values separated by commas"""
        if self.width is None:
            raise TypeError('Width is not initialised')
        else:
            return ','.join(str(width) for width in self.width)

    def _set_metad_params(self,
                          pace:        int,
                          width:       Union[Sequence[float], float],
                          height:      float,
                          biasfactor:  Optional[float] = None
                          ) -> None:
        """
        Define parameters used in well-tempered metadynamics.

        -----------------------------------------------------------------------
        Arguments:

            pace: (int) τ_G/dt, interval at which a new gaussian is placed

            width: (float) σ, standard deviation (parameter describing the
                           width) of the placed gaussian

            height: (float) ω, initial height of placed gaussians

            biasfactor: (float) γ, describes how quickly gaussians shrink,
                                larger values make gaussians to be placed
                                less sensitive to the bias potential
        """

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

        if height <= 0:
            raise ValueError('Gaussian height (ω) must be positive')

        else:
            self.height = height

        if biasfactor is not None and biasfactor < 1:
            raise ValueError('Biasfactor (γ) must be larger than one')

        else:
            self.biasfactor = biasfactor

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
                    self.setup.extend([line])

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

        return None

    def _from_atom_groups(self, name, atom_groups) -> None:
        """Generate DOFs from atom_groups"""

        self.name = name
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
                raise TypeError('Elements of atom_groups must all be sequences '
                                'or all be integers')

        else:
            raise TypeError('Atom groups are in incorrect format')

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
                        style:       str = 'scatter',
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
        hist = ax.hist2d(cvs_arrays[0], cvs_arrays[1], bins=300, density=True)

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
