import ase
import os
import shutil
import numpy as np
from time import time
from subprocess import Popen, PIPE
from scipy.spatial import distance_matrix
from mlptrain.box import Box
from mlptrain.log import logger
from mlptrain.config import Config
from mlptrain.potentials._base import MLPotential


class ACE(MLPotential):
    def _train(self) -> None:
        """
        Train this potential on the current training data by printing an
        appropriate 'input file' and running the Julia process
        """
        start_time = time()

        self._print_input(filename=f'{self.name}.jl')

        for config in self.training_data:
            if self.requires_non_zero_box_size and config.box is None:
                config.box = Box([100, 100, 100])

        self.training_data.save_xyz(filename=f'{self.name}_data.xyz')

        _check_julia_install_exists()

        logger.info(
            f'Training an ACE potential on *{len(self.training_data)}* '
            f'training data'
        )

        # Run the training using a specified number of total cores
        os.environ['JULIA_NUM_THREADS'] = str(Config.n_cores)

        p = Popen(
            [shutil.which('julia'), f'{self.name}.jl'],
            shell=False,
            stdout=PIPE,
            stderr=PIPE,
        )
        out, err = p.communicate(timeout=None)

        delta_time = time() - start_time
        logger.info(f'ACE training ran in {delta_time / 60:.1f} m')

        if any(
            (
                delta_time < 0.01,
                b'SYSTEM ABORT' in err,
                not os.path.exists(f'{self.name}.json'),
            )
        ):
            raise RuntimeError(f'ACE train errored with:\n{err.decode()}\n')

        for filename in (f'{self.name}_data.xyz', f'{self.name}.jl'):
            os.remove(filename)

        return None

    @property
    def requires_atomic_energies(self) -> bool:
        return True

    @property
    def requires_non_zero_box_size(self) -> bool:
        """ACE cannot use a zero size box"""
        return True

    @property
    def ase_calculator(self) -> 'ase.calculators.calculator.Calculator':
        """ASE calculator for this potential"""

        try:
            import pyjulip

        except ModuleNotFoundError:
            raise RuntimeError(
                'Failed to import pyjulip required for '
                'generating ASE calculators from ACE '
                'potentials.\n'
                'Install: https://github.com/casv2/pyjulip'
            )

        return pyjulip.ACE(f'./{self.name}.json')

    @property
    def _r_in_estimate(self) -> float:
        """
        Estimate the inner cut-off radius for the basis based on the minimum
        unique pairwise distance in all the configurations. Should be a little
        larger than that, so there is some data there

        -----------------------------------------------------------------------
        Returns:
            (float): r_min / Å
        """
        if len(self.training_data) == 0:
            raise ValueError('Cannot determine r_in. Had no training data')

        def pairwise_dists(_c):
            diag_shift = 9999.9 * np.eye(len(_c.coordinates))
            return distance_matrix(_c.coordinates, _c.coordinates) + diag_shift

        return (
            min(np.min(pairwise_dists(c)) for c in self.training_data) + 0.05
        )

    def _print_input(self, filename: str, **kwargs) -> None:
        """
        Print an input file appropriate for a ACE potential

        -----------------------------------------------------------------------
        Arguments:
            filename:

            **kwargs:
        """
        inp_file = open(filename, 'w')

        print(
            'using IPFitting, ACE, JuLIP, LinearAlgebra\n'
            'using JuLIP.MLIPs: combine, SumIP\n'
            'using ACE: z2i, i2z, order\n'
            f'BLAS.set_num_threads({Config.n_cores})\n',  # number of threads for the LSQ solver
            file=inp_file,
        )

        # first define the ACE basis specification
        _str = ', '.join([f':{s}' for s in self.system.unique_atomic_symbols])

        print(
            f'species = [{_str}]\n' 'N = 4',  # maximum correlation order
            file=inp_file,
        )

        for species in self.system.unique_atomic_symbols:
            print(f'z{species} = AtomicNumber(:{species})', file=inp_file)

        # maximum degrees for each correlation order
        print(
            'Dd = Dict("default" => 10,\n' '1 => 20,\n' '2 => 16,\n',
            file=inp_file,
        )

        for species in self.system.unique_atomic_symbols:
            if species == 'H':
                logger.warning('Not adding basis functions on H')

            print(
                f'(3, z{species}) => {16 if species != "H" else 0},',
                f'(4, z{species}) => {12 if species != "H" else 0},',
                file=inp_file,
            )

        print(')', file=inp_file)

        # for the basis function specified by (n, l)
        # degree = n_weight * n + l_weight * l
        # n_weights
        print(
            'Dn = Dict( "default" => 1.0 )\n'
            'Dl = Dict( "default" => 1.5 )',  # l_weights
            sep='\n',
            file=inp_file,
        )

        # r0 is a typical length scale for the distance transform
        print(
            'r0 = 1.3\n'
            f'r_in = {self._r_in_estimate:.4f}\n'  # inner cutoff of ACE, choose a little more than min dist in dataset
            'r_cut = 4.0\n'  # outer cutoff of ACE
            '\n'
            'deg_pair = 5\n'  # Specify the pair potential
            'r_cut_pair = 5.0\n',
            file=inp_file,
        )

        print('Vref = OneBody(', file=inp_file)
        for symbol in self.system.unique_atomic_symbols:
            print(
                f':{symbol} => {self.atomic_energies[symbol]},', file=inp_file
            )
        print(');', file=inp_file)

        # load the training data
        print(
            f'train_data = IPFitting.Data.read_xyz("{self.name}_data.xyz",\n'
            '                                     energy_key="energy",\n'
            '                                     force_key="forces",\n'
            '                                     virial_key="dummy");\n',
            file=inp_file,
        )

        # give weights for the different config_type-s
        print(
            'weights = Dict(\n'
            '       "default" => Dict("E" => 20.0, "F" => 1.0 , "V" => 0.0 )\n'
            '        );\n'
            'dbname = ""\n',  # change this to something to save the design matrix
            file=inp_file,
        )

        # specify the least squares solver, there are many implemented in IPFitting,
        # here are two examples with sensible defaults

        # Iterative LSQR with Laplacian scaling
        print(
            'damp = 0.1 # weight in front of ridge penalty, range 0.5 - 0.01\n'
            'rscal = 2.0 # power of Laplacian scaling of basis functions,  range is 1-4\n'
            'solver = (:itlsq, (damp, rscal, 1e-6, identity))\n'
            # simple riddge regression
            # r = 1.05 # how much of the training error to sacrifise for regularisation
            # solver = (:rid, r)
            f'save_name = "{filename.replace(".jl", ".json")}"\n',
            file=inp_file,
        )

        ######################################################################

        print(
            'Deg = ACE.RPI.SparsePSHDegreeM(Dn, Dl, Dd)\n'
            # construction of a basic basis for site energies
            'Bsite = rpi_basis(species = species,\n'
            '                   N = N,\n'
            '                   r0 = r0,\n'
            '                   D = Deg,\n'
            '                   rin = r_in, rcut = r_cut,\n'  # domain for radial basis (cf documentation)
            '                   maxdeg = 1.0,\n'  # maxdeg increases the entire basis size;
            '                   pin = 2)\n'  # require smooth inner cutoff
            # pair potential basis
            'Bpair = pair_basis(species = species, r0 = r0, maxdeg = deg_pair,\n'
            '                   rcut = r_cut_pair, rin = 0.0,\n'
            '                   pin = 0 )   # pin = 0 means no inner cutoff\n'
            'B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);\n'
            'println("The total number of basis functions is")\n'
            '@show length(B)\n'
            'dB = LsqDB(dbname, B, train_data);\n'
            'IP, lsqinfo = IPFitting.Lsq.lsqfit(dB, Vref=Vref,\n'
            '             solver=solver,\n'
            '             asmerrs=true, weights=weights)\n'
            'save_dict(save_name,'
            '           Dict("IP" => write_dict(IP), "info" => lsqinfo))\n'
            'rmse_table(lsqinfo["errors"])\n'
            'println("The L2 norm of the fit is ", round(norm(lsqinfo["c"]), digits=2))\n',
            file=inp_file,
        )

        inp_file.close()

        return None


def _check_julia_install_exists() -> None:
    """Ensure that a julia install is present"""

    if shutil.which('julia') is None:
        exit(
            "Failed to find a Julia installation. Make sure it's present "
            'in your $PATH'
        )
