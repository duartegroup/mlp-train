import os
import shutil
import numpy as np
from time import time
from subprocess import Popen, PIPE
from mltrain.potentials._base import MLPotential
from mltrain.config import Config
from mltrain.log import logger
from mltrain.utils import unique_filename


class NequIP(MLPotential):

    def _train(self) -> None:
        """
        Train a NeQUIP potential on a set of data. Requires an .npz file
        containing the coordinates, energies, forces and atomic numbers of
        each atom
        """
        start_time = time()

        self._print_input(filename=f'{self.name}.yml')
        self._print_training_npz(filename=f'{self.name}_data.npz')
        self._run_train()
        self._run_deploy()
        self._clean_up_dirs()

        delta_time = time() - start_time
        logger.info(f'NeQUIP training ran in {delta_time / 60:.1f} m')

        return None

    @property
    def ase_calculator(self) -> 'ase.calculators.calculator.Calculator':
        """
        Instance of an ASE calculator for a NequIP potential

        ----------------------------------------------------------------------
        Returns:
            (ase.Calculator):
        """

        try:
            from nequip.dynamics.nequip_calculator import NequIPCalculator

        except ModuleNotFoundError:
            raise ModuleNotFoundError('NeQUIP install not found, install it '
                                      'here: https://github.com/mir-group/nequip')

        calculator = NequIPCalculator.from_deployed_model(
                          f'{self.name}_deployed.pth',
                          device='cpu'
                    )

        return calculator

    @property
    def requires_atomic_energies(self) -> bool:
        """NeQUIP doesn't require atomic energies in the dataset"""
        return False

    @property
    def requires_non_zero_box_size(self) -> bool:
        """NeQUIP can train on vacuum data without PBC"""
        return False

    def _print_training_npz(self, filename):
        """Print a compressed numpy data file of the required data"""

        atomic_numbers = []
        energies = []
        forces = []
        coords = []

        for config in self.training_data:
            z = [atom.atomic_number for atom in config.atoms]
            e = [config.energy.true]
            f = config.forces.true
            r = config.coordinates

            atomic_numbers = np.array(z)
            energies.append(np.array(e))
            forces.append(np.array(f))
            coords.append(np.array(r))

        np.savez(filename, Z=atomic_numbers, E=energies, F=forces, R=coords)

    def _print_input(self, filename):
        """Print a .yml file suitable for a NeQUIP training"""

        yml_file = open(filename, 'w')

        train_frac = Config.nequip_params["train_fraction"]
        if train_frac >= 1 or train_frac <= 0:
            raise RuntimeError('Cannot train on a training fraction ∉ [0, 1]')

        print(
            'root: ./',
            f'run_name: {self.name}',
            'seed: 0 ',
            'restart: false  ',
            'append: false ',
            'default_dtype: float32  ',
            f'r_max: {Config.nequip_params["cutoff"]}',
            'num_layers: 6 ',
            'chemical_embedding_irreps_out: 32x0e',
            'feature_irreps_hidden: 32x0o + 32x0e + 16x1o + 16x1e + 8x2o + 8x2e ',
            'irreps_edge_sh: 0e + 1o + 2e  ',
            'conv_to_output_hidden_irreps_out: 16x0e',
            'nonlinearity_type: gate',
            'resnet: false ',
            'num_basis: 8 ',
            'invariant_layers: 2  ',
            'invariant_neurons: 64 ',
            'avg_num_neighbors: null  ',
            'use_sc: true  ',
            'dataset: npz  # type of data set, can be npz or ase',
            f'dataset_file_name: ./{self.name}_data.npz ',
            'key_mapping:',
            '  Z: atomic_numbers  # atomic species, integers',
            '  E: total_energy    # total potential eneriges to train to',
            '  F: forces          # atomic forces to train to',
            '  R: pos             # raw atomic positions',
            'npz_fixed_field_keys: ',
            '  - atomic_numbers',
            'wandb: false  ',
            f'n_train: {int(train_frac * len(self.training_data))}',
            f'n_val: {int((1. - train_frac) * len(self.training_data))}',
            'learning_rate: 0.01 ',
            'batch_size: 5  ',
            'max_epochs: 1000  ',
            'metrics_key: loss ',
            'use_ema: false ',
            'ema_decay: 0.999  ',
            'loss_coeffs: ',
            '  forces: 100  ',
            '  total_energy: 1 ',
            'metrics_components:',
            '  - - forces  # key',
            '    - rmse  # "rmse" or "mse"',
            '    - PerSpecies: True  ',
            '      report_per_component: False',
            '  - - forces',
            '    - mae',
            '    - PerSpecies: True',
            '      report_per_component: False',
            '  - - total_energy',
            '    - mae',
            'early_stopping_lower_bounds:',
            '   e_mae: 0.02  # ~0.5 kcal mol-1',
            'optimizer_name: Adam',
            'optimizer_amsgrad: true',
            'lr_scheduler_name: ReduceLROnPlateau',
            'lr_scheduler_patience: 100',
            'lr_scheduler_factor: 0.5',
            sep='\n', file=yml_file)

        yml_file.close()
        return None

    def _run_train(self):
        """Run the training, once the input file(s) have been generated"""

        train_executable_path = shutil.which('nequip-train')

        if train_executable_path is None:
            raise RuntimeError('No NeQUIP install found!')

        logger.info(f'Training a NeQUIP potential on '
                    f'*{len(self.training_data)}* training data')

        p = Popen([train_executable_path, f'{self.name}.yml'],
                  shell=False,
                  stdout=PIPE,
                  stderr=PIPE,
                  env={**os.environ, 'OMP_NUM_THREADS': str(Config.n_cores)})
        out, err = p.communicate(timeout=None)

        if b'SYSTEM ABORT' in err or b'raise' in err:
            raise RuntimeError(f'NeQUIP train errored with:\n{err.decode()}\n')

        return None

    def _run_deploy(self):
        """Deploy a NeQUIP model"""
        logger.info(f'Deploying a NeQUIP potential')

        p = Popen([shutil.which('nequip-deploy'), 'build', f'{self.name}/',
                   f'{self.name}_deployed.pth'],
                  shell=False,
                  stdout=PIPE,
                  stderr=PIPE)
        _, _ = p.communicate(timeout=None)

        return None

    def _clean_up_dirs(self):
        """Clean up the directories created by NeQUIP train"""

        shutil.rmtree('processed')
        shutil.make_archive(unique_filename(f'{self.name}.zip')[:-4],
                            'zip',
                            self.name)
        try:
            shutil.rmtree(self.name)
        except OSError:
            logger.warning(f'Failed to remove {self.name}/')

        return None
