import mlptrain as mlt
import argparse
from mlptrain.config import Config
from mlptrain.potentials._base import MLPotential
import os
import time
import numpy as np
import logging
from mlptrain.log import logger
import autode as ade
from typing import Optional
import ase

try:
    from mace.calculators import MACECalculator
    from mace.cli.run_train import run as train_mace
    from mace import tools
    import torch
    import gc
except ModuleNotFoundError:
    pass


class MACE(MLPotential):
    def __init__(
        self,
        name: str,
        system: 'mlt.System',
        foundation: Optional[str] = None,
    ) -> None:
        """
        MACE machine learning potential

        -----------------------------------------------------------------------
        Arguments:

            name: (str) Name of the potential, used in naming output files

            system: (mlptrain.System) Object defining the system without
                                      specifying the coordinates

            foundation: (str) Name of the foundation model used in fine-tunning
                         like "medium_off" for MACE-OFF(M), "medium" for MACE-MP-0(M).
                         Here, only naive fine-tuning is supported.
                         More details on https://github.com/ACEsuit/mace/tree/main?tab=readme-ov-file#pretrained-foundation-models
        """
        super().__init__(name=name, system=system)

        try:
            import mace
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'MACE install not found, install it '
                'here: https://github.com/ACEsuit/mace'
            )

        self.foundation = foundation
        logging.info(f'MACE version: {mace.__version__}')

        tools.set_seeds(345)
        tools.set_default_dtype(str(Config.mace_params['dtype']))

    @property
    def filename(self) -> str:
        """Name of the file where potential is stored"""
        return f'{self.name}.model'

    @property
    def requires_atomic_energies(self) -> bool:
        return True

    @property
    def requires_non_zero_box_size(self) -> bool:
        """MACE cannot use a zero size box"""
        return True

    @property
    def get_E0s(self):
        E0s_dictionary = {}
        atomic_energies = self.atomic_energies
        for key, value in atomic_energies.items():
            Atom = ade.Atom(atomic_symbol=key)
            atomic_number = Atom.atomic_number
            E0s_dictionary[atomic_number] = float(value)
        return E0s_dictionary

    @property
    def valid_fraction(self) -> float:
        """Fraction of the whole dataset to be used as validation set"""
        _min_dataset = -(1 // -Config.mace_params['valid_fraction'])

        if self.n_train == 1:
            raise ValueError(
                'MACE training requires at least ' '2 configurations'
            )
        elif self.n_train >= _min_dataset:
            return Config.mace_params['valid_fraction']
        else:
            # Valid fraction which sets at least 1 datapoint for validation
            _unrounded_valid_fraction = 1 / self.n_train
            return -((_unrounded_valid_fraction * 100) // -1) / 100

    @property
    def batch_size(self) -> int:
        """Batch size of the training set"""
        if (
            self.n_train * (1 - Config.mace_params['valid_fraction'])
            < Config.mace_params['batch_size']
        ):
            return int(
                np.floor(
                    self.n_train * (1 - Config.mace_params['valid_fraction'])
                )
            )
        else:
            return Config.mace_params['batch_size']

    @property
    def args(self) -> 'argparse.Namespace':
        """Namespace containing mostly default MACE parameters"""

        args_list = [
            '--name',
            self.name,
            '--max_L',
            str(Config.mace_params['max_L']),
            '--train_file',
            f'{self.name}_data.xyz',
            '--valid_fraction',
            str(self.valid_fraction),
            '--energy_weight',
            str(Config.mace_params['energy_weight']),
            '--forces_weight',
            str(Config.mace_params['forces_weight']),
            '--config_type_weights',
            str(Config.mace_params['config_type_weights']),
            '--E0s',
            str(self.get_E0s),
            '--model',
            str(Config.mace_params['model']),
            '--hidden_irreps',
            str(Config.mace_params['hidden_irreps']),
            '--r_max',
            str(Config.mace_params['r_max']),
            '--lr',
            str(Config.mace_params['lr']),
            '--scaling',
            'rms_forces_scaling',
            '--batch_size',
            str(self.batch_size),
            '--valid_batch_size',
            str(self.batch_size),
            '--max_num_epochs',
            str(Config.mace_params['max_num_epochs']),
            '--error_table',
            str(Config.mace_params['error_table']),
            '--loss',
            str(Config.mace_params['loss']),
            '--correlation',
            str(Config.mace_params['correlation']),
            '--scheduler_patience',
           str(Config.mace_params['scheduler_patience']),
            '--patience',
            str(Config.mace_params['patience']),
            '--start_swa',
            str(Config.mace_params['start_swa']),
            '--swa',
            '--ema',
            '--ema_decay',
            str(0.999),
            '--device',
            str(Config.mace_params['device']),
            '--default_dtype',
            str(Config.mace_params['dtype']),
            '--seed',
            str(Config.mace_params['seed']),
            '--energy_key',
            'energy',
            '--forces_key',
            'forces',
            '--num_workers',
            str(Config.mace_params['num_workers']),
        ]

        if self.foundation is not None:
            args_list.append('--foundation_model')
            args_list.append(f'{self.foundation}')

        if Config.mace_params['save_cpu'] is True:
            args_list.append('--save_cpu')

        if Config.mace_params['restart_latest'] is True:
            args_list.append('--restart_latest')

        args = tools.build_default_arg_parser().parse_args(args_list)
        return args

    @property
    def ase_calculator(self) -> 'ase.calculators.calculator.Calculator':
        """ASE calculator for MACE potential"""

        calculator = MACECalculator(
            model_paths=self.filename,
            device=Config.mace_params['calc_device'],
        )
        return calculator

    def _train(self, n_cores: Optional[int] = None) -> None:
        """
        Train a MACE potential using the data as .xyz file and save the
        final potential as .model file

        -----------------------------------------------------------------------
        Arguments:

            n_cores: (int) Number of cores to use in training
        """

        n_cores = n_cores if n_cores is not None else Config.n_cores
        os.environ['OMP_NUM_THREADS'] = str(n_cores)
        logger.info(
            'Training a MACE potential on '
            f'*{len(self.training_data)}* training data, '
            f'using {n_cores} cores for training.'
        )

        for config in self.training_data:
            if self.requires_non_zero_box_size and config.box is None:
                config.box = mlt.box.Box([100, 100, 100])

        self.training_data.save_xyz(filename=f'{self.name}_data.xyz')

        start_time = time.perf_counter()

        train_mace(self.args)

        delta_time = time.perf_counter() - start_time

        logger.info(f'MACE training ran in {delta_time / 60:.1f} m.')

        os.remove(f'{self.name}_data.xyz')

        gc.collect()
        torch.cuda.empty_cache()

        return None
