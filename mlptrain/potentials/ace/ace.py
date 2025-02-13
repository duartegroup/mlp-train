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
            encoding='utf-8',
            stdout=PIPE,
            stderr=PIPE,
        )
        out, err = p.communicate(timeout=None)

        filename_ace_out = 'ACE_output.out'

        with open(filename_ace_out, 'a') as f:
            f.write(f'ACE training output:\n{out}')
            if err:
                f.write(f'ACE training error:\n{err}')

        delta_time = time() - start_time
        logger.info(f'ACE training ran in {delta_time / 60:.1f} m')

        if any(
            (
                delta_time < 0.01,
                'SYSTEM ABORT' in err,
                p.returncode != 0,
                not os.path.exists(f'{self.name}.json'),
            )
        ):
            raise RuntimeError(
                f'ACE train errored with a return code:\n{p.returncode}\n'
                f'and error:\n{err}\n'
            )

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
            'using ExtXYZ\n' 'using ACEpotentials\n',
            file=inp_file,
        )

        # first define the ACE basis specification
        _str = ', '.join([f':{s}' for s in self.system.unique_atomic_symbols])

        print(
            f'species = [{_str}]\n'
            f"correlation_order = {Config.ace_params['correlation_order']}\n"  # maximum correlation order (body order - 1)
            f"total_degree = {Config.ace_params['total_degree']}",  # maximum total polynomial degree used for the basis
            file=inp_file,
        )

        # r0 is a typical length scale for the distance transform
        print(
            f"r_cut = {Config.ace_params['r_cut']}\n"  # outer cutoff of ACE
            '\n',
            file=inp_file,
        )

        print('Eref = Dict(', file=inp_file)
        for symbol in self.system.unique_atomic_symbols:
            print(
                f':{symbol} => {self.atomic_energies[symbol]},', file=inp_file
            )
        print(');', file=inp_file)

        # load the training data
        print(
            f'data_file = "{self.name}_data.xyz"\n'
            f'data_set = ExtXYZ.load(data_file)\n'
            f'data_keys = (energy_key = "energy", force_key = "forces", virial_key = "dummy")\n',
            file=inp_file,
        )

        # give weights for the different config_type-s
        print(
            'weights = Dict(\n'
            '       "default" => Dict("E" => 20.0, "F" => 1.0 , "V" => 0.0 )\n'
            '        );\n',
            file=inp_file,
        )

        print(
            'model = acemodel(elements = species,\n'
            '                   rcut = r_cut,\n'
            '                   order = correlation_order,\n'
            '                   totaldegree = total_degree,\n'
            '                   Eref = Eref\n',
            '                   );',
            file=inp_file,
        )

        # Available solvers: QR, BLR, LSQR, RRQR,
        # information can be found here:

        if Config.ace_params['solver'] == 'QR':
            print('solver = ACEfit.QR(lambda = 1e-1)\n', file=inp_file)
        elif Config.ace_params['solver'] == 'LSQR':
            print(
                'solver = ACEfit.LSQR(damp = 1e-4, atol = 1e-6)\n',
                file=inp_file,
            )
        else:
            raise NotImplementedError(
                'The solver is not supported. Available options are QR or LSQR.'
            )

        print(
            'acefit!(model, data_set;\n'
            '        solver = solver, data_keys...);\n',
            file=inp_file,
        )

        print(
            '@info("Training Error Table")\n'
            'ACEpotentials.linear_errors(data_set, model; weights=weights);\n',
            file=inp_file,
        )

        print(
            f'save_potential("{self.name}.json", model)',
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
