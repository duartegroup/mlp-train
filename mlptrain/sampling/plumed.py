from typing import Sequence, Optional, Union


class PlumedBias:
    """Class for storing collective variables and parameters used in biased
    simulations"""

    def __init__(self,
                 cvs: Union[Sequence['_PlumedCV'],
                                     '_PlumedCV'] = None,
                 file_name: str = None):
        """
        Class for storing collective variables and parameters used in biased
        simulations, parameters are not initialised with the object and have
        to be defined seperately using PlumedBias methods.

        Can be initialised from a complete PLUMED input file as well.

        -----------------------------------------------------------------------
        Arguments:

            cvs: Sequence of PLUMED collective variables

            file_name (str): Complete PLUMED input file
        """

        self.md_method = None
        self.setup = None
        self.pace = self.width = self.height = self.biasfactor = None

        if file_name is not None:
            self._from_file(file_name)

        elif cvs is not None:

            # e.g. cvs = [cv1, cv2]; (cv1, cv2)
            if isinstance(cvs, list) or isinstance(cvs, tuple):
                self.cvs = cvs

            # e.g. cvs = cv1
            elif cvs.__class__.__base__ == _PlumedCV:
                self.cvs = [cvs]

            else:
                raise TypeError('Supplied CVs are in incorrect format')

        else:
            raise TypeError('PLUMED bias instantiation requires '
                            'a list of collective variables (CVs) '
                            'or a file containing PLUMED-type input')

    def set_metad_params(self,
                         pace: int,
                         width: float,
                         height: float,
                         biasfactor: float) -> None:
        """
        Define parameters used in well-tempered metadynamics.

        -----------------------------------------------------------------------
        Arguments:

            pace (int): τ_G/dt, interval at which a new gaussian is placed

            width (float): σ, standard deviation (parameter describing the
                           width) of the placed gaussian

            height (float): ω, initial height of placed gaussians

            biasfactor (float): γ, describes how quickly gaussians shrink,
                                larger values make gaussians to be placed
                                less sensitive to the bias potential
        """

        self.md_method = 'metadynamics'

        if not isinstance(pace, int) or pace <= 0:
            raise ValueError('Pace (τ_G/dt) must be a positive integer')
        else:
            self.pace = pace

        if width <= 0:
            raise ValueError('Gaussian width (σ) must be positive')
        else:
            self.width = width

        if height <= 0:
            raise ValueError('Gaussian height (ω) must be positive')
        else:
            self.height = height

        if biasfactor < 1:
            raise ValueError('Bias factor (γ) must be larger than one')
        else:
            self.biasfactor = biasfactor

    @property
    def cv_sequence(self) -> str:
        """String containing names of collective variables separated
        by commas"""
        cv_names = (cv.name for cv in self.cvs)
        return ','.join(cv_names)

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
                 name: str = None,
                 atom_groups: Sequence = None,
                 file_name: str = None,
                 component: Optional[str] = None):
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

            name (str): Name of the collective variable (only for generating
                        CVs from atom_groups, the name when generating CVs from
                        a file is taken directly from the file)

            atom_groups (Sequence[Sequence[int]]): List of atom index sequences
                                                which are used to generate DOFs,
                                                e.g. [(0, 1), (2, 3)];
                                                     [0, 1, 2]

            file_name (str): Name of the PLUMED file used to generate a CV
                             from that file

            component (str): Name of a component of the last CV in the supplied
                             PLUMED input file to use as a collective variable,
                             e.g. 'spath' for PATH collective variable.
        """

        self.name = None
        self.dof_names = None
        self.setup = []

        if file_name is not None:
            self._from_file(file_name, component)

        elif atom_groups is not None:
            self._from_atom_groups(name, atom_groups)

        else:
            raise TypeError('Collective variable instantiation requires '
                            'groups of atom indices (DOFs) '
                            'or a file containing PLUMED-type input')

    def _from_file(self, file_name, component) -> None:
        """Method to generate DOFs and a CV from a file"""

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
        """Method to generate DOFs from atom_groups"""

        self.name = name
        self.dof_names = []

        # e.g. atom_groups = (2)
        if (not isinstance(atom_groups, tuple)
                and not isinstance(atom_groups, list)):

            raise TypeError('atom_groups must be a tuple or a list')

        # atom_groups = []; ()
        elif len(atom_groups) == 0:

            raise TypeError('atom_groups cannot be an empty list '
                            'or an empty tuple')

        # e.g. atom_groups = [(1, 2), (3, 4)]; ([0, 1])
        elif (all(isinstance(atom_group, tuple) or isinstance(atom_group, list)
                  for atom_group in atom_groups)):

            for idx, atom_group in enumerate(atom_groups):
                self._atom_group_to_dof(idx, atom_group)

        # e.g. atom_groups = [0, 1]
        elif all(isinstance(atom_index, int) for atom_index in atom_groups):

            self._atom_group_to_dof(0, atom_groups)

        # e.g. atom_groups = [(1, 2, 3), 1]
        else:
            raise TypeError('Elements of atom_groups must all be sequences '
                            'or all be integers')

        return None

    def _atom_group_to_dof(self, idx, atom_group) -> None:
        """Method to check the atom group and generate a DOF"""

        # PLUMED atom enumeration starts from 1
        atom_list = [f'{i + 1}' for i in atom_group]
        atoms = ','.join(atom_list)

        if len(atom_list) < 2:
            raise ValueError('Atom group must contain at least two atoms')

        if len(atom_list) == 2:
            dof_name = f'{self.name}_dist{idx + 1}'
            self.dof_names.append(dof_name)
            self.setup.extend([f'{dof_name}: '
                               f'DISTANCE ATOMS={atoms}'])

        if len(atom_list) == 3:
            dof_name = f'{self.name}_ang{idx + 1}'
            self.dof_names.append(dof_name)
            self.setup.extend([f'{dof_name}: '
                               f'ANGLE ATOMS={atoms}'])

        if len(atom_list) == 4:
            dof_name = f'{self.name}_tor{idx + 1}'
            self.dof_names.append(dof_name)
            self.setup.extend([f'{dof_name}: '
                               f'TORSION ATOMS={atoms}'])

        if len(atom_list) > 4:
            raise NotImplementedError('Instatiation using atom groups '
                                      'is only implemented for groups '
                                      'not larger than four')

        return None

    @property
    def dof_sequence(self) -> str:
        """String containing names of DOFs separated by commas"""
        return ','.join(self.dof_names)


class PlumedAverageCV(_PlumedCV):
    """Class used to initialise a PLUMED collective variable as an average
    between multiple degrees of freedom"""

    def __init__(self,
                 name: str,
                 atom_groups: Sequence = None):
        """
        PLUMED collective variable as an average between multiple degrees of
        freedom (distances, angles, torsions),

        e.g. [(0, 1), (2, 3), (4, 5)] gives ζ = 1/3 * (r_01 + r_23 + r_45)

        -----------------------------------------------------------------------
        Arguments:

            name (str): Name of the collective variable

            atom_groups (Sequence[Sequence[int]]): List of atom index sequences
                                                which are used to generate DOFs
        """

        super().__init__(name=name,
                         atom_groups=atom_groups)

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
                 name: str,
                 atom_groups: Sequence = None):
        """
        PLUMED collective variable as a difference between two degrees of
        freedom (distances, angles, torsions),

        e.g. [(0, 1), (1, 2)] gives ζ = r_12 - r_01

        -----------------------------------------------------------------------
        Arguments:

            name (str): Name of the collective variable

            atom_groups (Sequence[Sequence[int]]): List of atom index sequences
                                                which are used to generate DOFs
        """

        super().__init__(name=name,
                         atom_groups=atom_groups)

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
                 file_name: str,
                 component: Optional[str] = None):
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

            file_name (str): Name of the PLUMED file used to generate a CV
                             from that file

            component (str): Name of a component of the last CV in the supplied
                             PLUMED input file to use as a collective variable
        """
        super().__init__(file_name=file_name,
                         component=component)
