import os
import shutil
from typing import Optional
from copy import deepcopy
from subprocess import Popen, PIPE
from time import time
from autode.atoms import Atom
from mltrain.config import Config
from mltrain.log import logger
from mltrain.potentials._base import MLPotential


class GAP(MLPotential):

    def __init__(self,
                 name:           str,
                 system:         Optional['mltrain.System'] = None,
                 default_params: bool = True):
        """
        Gaussian Approximation Potential. Parameters default to using all
        unique pairs of SOAPs

        -----------------------------------------------------------------------
        Arguments:
            name:

            system: System defining the atoms, so elements to include in the
                    parameters (SOAPs)

            default_params: Whether to use default parameters
        """
        super().__init__(name=name if not name.endswith('.xml') else name[:-4],
                         system=system)

        self.params = None

        if system is not None and default_params:
            self.params = _GAPParameters(atoms=system.atoms)

    @property
    def requires_atomic_energies(self) -> bool:
        return False

    @property
    def requires_non_zero_box_size(self) -> bool:
        """GAP can use a zero size box"""
        return False

    @property
    def xml_filename(self):
        return f'{self.name}.xml'

    def _check_xml_exists(self):
        """Raise an exception if the parameter file (.xml) doesn't exist"""
        if not os.path.exists(self.xml_filename):
            raise IOError(f'GAP parameter file ({self.xml_filename}) did not '
                          f'exist')

    @property
    def ase_calculator(self):
        """
        ASE Calculator instance to evaluate the energy using a GAP with
        parameter filename: self.xml_filename

        :return: (ase.Calculator)
        """
        try:
            import quippy
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Quippy was not installed. Try\n'
                                      'pip install quippy-ase')

        self._check_xml_exists()
        return quippy.potential.Potential("IP GAP",
                                          param_filename=self.xml_filename)

    @property
    def _train_command(self):
        """Generate the teach_sparse function call for this system of atoms"""

        general = self.params.general
        params = ('default_sigma={'
                  f'{general["sigma_E"]:.6f} {general["sigma_F"]:.6f} 0.0 0.0'
                  '} ')

        params += 'e0_method=average gap={'

        # Likewise with all the SOAPs to be added
        for symbol, soap in self.params.soap.items():
            logger.info(f'Adding SOAP:              {symbol}')
            other_atomic_ns = [Atom(s).atomic_number for s in soap["other"]]
            logger.info(f'with neighbours           {soap["other"]}')

            params += ('soap sparse_method=cur_points '
                       f'n_sparse={int(soap["n_sparse"])} '
                       f'covariance_type=dot_product '
                       f'zeta=4 '
                       f'atom_sigma={soap["sigma_at"]} '
                       f'cutoff={soap["cutoff"]} '
                       f'delta=1.0 '
                       f'add_species=F '
                       f'n_Z=1 '
                       f'n_species={len(soap["other"])} '
                       'species_Z={{'
                       # Remove the brackets from the ends of the list
                       f'{str(other_atomic_ns)[1:-1]}'
                       '}} '
                       f'Z={Atom(symbol).atomic_number} '
                       f'n_max={int(2 * soap["l_max"])} '
                       f'l_max={int(soap["l_max"])}: ')

        # Remove the final unnecessary colon
        params = params.rstrip(': ')

        # Reference energy and forces labels and don't separate xml files
        params += ('} energy_parameter_name=energy '
                   'force_parameter_name=forces '
                   'sparse_separate_file=F')

        # GAP needs the training data, some parameters and a file to save to
        return [f'at_file={self.name}_data.xyz', params, f'gp_file={self.name}.xml']

    def _train(self):
        """Train this GAP on its training data"""

        if self.params is None or len(self.params.soap) == 0:
            raise RuntimeError(f'Cannot train a GAP({self.name}) - had no '
                               f'parameters')

        if shutil.which('gap_fit') is None:
            raise RuntimeError('Cannot train a GAP without a gap_fit '
                               'executable present')

        logger.info('Training a Gaussian Approximation potential on '
                    f'*{len(self.training_data)}* training data points')

        start_time = time()

        self.training_data.save_xyz(filename=f'{self.name}_data.xyz')

        # Run the training using a specified number of total cores
        os.environ['OMP_NUM_THREADS'] = str(Config.n_cores)

        p = Popen([shutil.which('gap_fit')] + self._train_command,
                  shell=False,
                  stdout=PIPE,
                  stderr=PIPE)
        out, err = p.communicate()

        delta_time = time() - start_time
        logger.info(f'GAP training ran in {delta_time/60:.1f} m')

        if any((delta_time < 0.01,
                b'SYSTEM ABORT' in err,
                not os.path.exists(f'{self.name}.xml'))):

            raise RuntimeError(f'GAP train errored with:\n '
                               f'{err.decode()}\n'
                               f'{" ".join(self._train_command)}')

        os.remove(f'{self.name}_data.xyz.idx')

        return None


class _GAPParameters:

    @staticmethod
    def _soap_dict(atom_symbols):
        """Set the SOAP parameters"""
        soap_dict, added_pairs = {}, []

        for symbol in set(atom_symbols):

            if symbol == 'H':
                logger.warning('Not adding SOAP on H')
                continue

            params = deepcopy(Config.gap_default_soap_params)

            # Add all the atomic symbols that aren't this one, the neighbour
            # density for which also hasn't been added already
            params["other"] = [s for s in set(atom_symbols)
                               if s+symbol not in added_pairs
                               and symbol+s not in added_pairs]

            # If there are no other atoms of this type then remove the self
            # pair
            if atom_symbols.count(symbol) == 1:
                params["other"].remove(symbol)

            for other_symbol in params["other"]:
                added_pairs.append(symbol+other_symbol)

            if len(params["other"]) == 0:
                logger.info(f'Not adding SOAP to {symbol} - should be covered')
                continue

            soap_dict[symbol] = params

        return soap_dict

    def __init__(self, atoms):
        """
        Parameters for a GAP potential

        Arguments:
             atoms: Atoms used to to generate parameters for a GAP potential
        """

        self.general = deepcopy(Config.gap_default_params)
        self.soap = self._soap_dict(atom_symbols=[a.label for a in atoms])
