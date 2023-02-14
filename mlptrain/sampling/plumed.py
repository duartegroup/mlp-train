import mlptrain
from typing import Sequence


class PlumedBias:
    """Class for storing collective variables and parameters used in biased
    simulations"""

    def __init__(self,
                 cvs: Sequence['mlptrain.sampling.plumed._PlumedCV']):
        """
        Class for storing collective variables and parameters used in biased
        simulations, parameters are not initialised with the object and have
        to be defined seperately using PlumedBias methods.

        -----------------------------------------------------------------------
        Arguments:

            cvs (Sequence): Sequence of PLUMED collective variable objects
        """

        self.cvs = cvs
        self.pace = self.width = self.height = self.biasfactor = None

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


class _PlumedCV:
    """Parent class containing methods for initialising PLUMED collective
    variables"""

    def __init__(self,
                 name: str = None,
                 atom_groups: Sequence = None,
                 file_name: str = None):
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
                                                e.g. [(0, 1), (2, 3)]

            file_name (str): Name of the PLUMED file used to generate a CV
                             from that file
        """

        self.name = None
        self.dof_names, self.setup = [], []

        if file_name is not None:
            self._from_file(file_name)

        elif atom_groups is not None:
            self._from_atom_groups(name, atom_groups)

        else:
            raise ValueError('Collective variable instantiation requires '
                             'groups of atom indices (DOFs) '
                             'or a file containing PLUMED-type input')

    def _from_file(self, file_path) -> None:
        """Method to generate DOFs and a CV from a file"""

        with open(file_path, 'r') as f:
            for line in f:
                if line[0] == '#' or line == '\n':
                    pass
                else:
                    line = line.strip()
                    self.setup.extend([line])

        _names = [line.split(':')[0] for line in self.setup]
        _actions = [line.split(' ')[1] for line in self.setup]
        _names_and_actions = list(zip(_names, _actions))

        self.name = _names_and_actions.pop()[0]

        for name, action in _names_and_actions:
            if action != 'GROUP':
                self.dof_names.append(f'{self.name}_{name}')

        return None

    def _from_atom_groups(self, name, atom_groups) -> None:
        """Method to generate DOFs from atom_groups"""

        self.name = name

        for idx, atom_group in enumerate(atom_groups):

            # PLUMED atom enumeration starts from 1
            atom_tuple = (f'{i+1}' for i in atom_group)
            atoms = ','.join(atom_tuple)

            if len(atom_group) < 2:
                raise ValueError('Atom group must contain at least two atoms')

            if len(atom_group) == 2:
                dof_name = f'{self.name}_dist{idx}'
                self.dof_names.append(dof_name)
                self.setup.extend([f'{dof_name}: '
                                   f'DISTANCE ATOMS={atoms}'])

            if len(atom_group) == 3:
                dof_name = f'{self.name}_ang{idx}'
                self.dof_names.append(dof_name)
                self.setup.extend([f'{dof_name}: '
                                   f'ANGLE ATOMS={atoms}'])

            if len(atom_group) == 4:
                dof_name = f'{self.name}_tor{idx}'
                self.dof_names.append(dof_name)
                self.setup.extend([f'{dof_name}: '
                                   f'TORSION ATOMS={atoms}'])

            if len(atom_group) > 4:
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
        func = f'{1/len(self.dof_names)}*({dof_sum})'

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
                 file_name: str):
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
        """
        super().__init__(file_name=file_name)
