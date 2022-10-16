import numpy as np
from time import time
from copy import deepcopy
from mlptrain.box import Box
from mlptrain.log import logger
from mlptrain.config import Config
from mlptrain.potentials._base import MLPotential
import ast
import logging
import os
from typing import Optional
import torch.nn.functional
from e3nn import o3
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage
from mlptrain.config import Config
from ase.data import chemical_symbols
import dataclasses
from prettytable import PrettyTable
import torch


@dataclasses.dataclass
class SubsetCollection:
    try:
        from mace import data

    except ModuleNotFoundError:
        raise ModuleNotFoundError('MACE install not found, install it '
                                      'here: https://github.com/ACEsuit/mace')
    else:
        train: data.Configurations
        valid: data.Configurations

class MACE(MLPotential):

    def _train_file(self,
                    valid_fraction = 0.1,
                    config_type_weights = '{"Default":1.0}' ,
                    model = 'MACE',
                    loss = 'weighted',
                    hidden_irreps = '128x0e + 128x1o',
                    batch_size = 10,
                    r_max=5 ,
                    correlation = 3,
                    device = 'cuda',
                    error_table = 'TotalMAE',
                    swa = True,
                    #start_swa= 1200,
                    ema = False,
                    ema_decay=0.99,
                    amsgrad = True,
                    #max_num_epochs=800,
                    restart_latest = True,
                    save_cpu = True
                    ) -> None:
        """
        Train MACE potential is adjusted from run_train.py in mace package
        https://github.com/ACEsuit/mace/tree/main/scripts
        """
        try:
            import mace
            from mace import data, modules, tools
            from mace.tools import torch_geometric

        except ModuleNotFoundError:
            raise ModuleNotFoundError('MACE install not found, install it '
                                      'here: https://github.com/ACEsuit/mace')
            
            
        if len(self.training_data) <100:
            max_num_epochs = 1000
        elif 100<=len(self.training_data) <300:
            max_num_epochs = 1200
        else:
            max_num_epochs = 1500

        args = tools.build_default_arg_parser().parse_args(['--name', self.name, '--train_file', f'{self.name}_data.xyz', '--default_dtype', 'float64' ])
        logging.info(f"{args.train_file}")
        tag = self.name
      
        # Setup
        tools.set_seeds(args.seed)
        tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
        try:
            logging.info(f"MACE version: {mace.__version__}")
        except AttributeError:
            logging.info("Cannot find MACE version, please install MACE via pip")
        #logging.info(f"Configuration: {args}")
        device = tools.init_device(device_str = device)
        tools.set_default_dtype(args.default_dtype)

        try:
            config_type_weights = ast.literal_eval(config_type_weights)
            assert isinstance(config_type_weights, dict)
        except Exception as e:  # pylint: disable=W0703
            logging.warning(
               f"Config type weights not specified correctly ({e}), using Default"
            )
            config_type_weights = {"Default": 1.0}

        # Data preparation
        collections = self.get_dataset_from_xyz(train_path=args.train_file,
                                                valid_path=args.valid_file,
                                                valid_fraction=valid_fraction,
                                                config_type_weights=config_type_weights,
                                                seed=args.seed,
                                                energy_key=args.energy_key,
                                                forces_key=args.forces_key)

        atomic_energies_dict = self.system.atomic_energies

        logging.info(
        f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}"
        )
        # Atomic number table
        # yapf: disable
        z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
        )
        # yapf: enable
        logging.info(z_table)

        z_table_symbol = [ chemical_symbols[i] for i in z_table.zs]
        logging.info(z_table_symbol)

        if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
            raise RuntimeError(
                "atomic energies needed"
                )
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table_symbol]
        )
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")
        
        train_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
                for config in collections.train
            ],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        valid_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
                for config in collections.valid
            ],
            batch_size=args.valid_batch_size,
            shuffle=False,
            drop_last=False,
        )

        loss_fn: torch.nn.Module
        if loss == "weighted":
            loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
            )
        elif loss == "forces_only":
            loss_fn = modules.WeightedForcesLoss(forces_weight=args.forces_weight)
        else:
            loss_fn = modules.EnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
            )
        logging.info(loss_fn)
        
        if args.compute_avg_num_neighbors:
            args.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
        logging.info(f"Average number of neighbors: {args.avg_num_neighbors:.3f}")

        # Build model
        logging.info("Building model")
        model_config = dict(
        r_max=r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
        )

        model: torch.nn.Module

        if model == "MACE":
            if args.scaling == "no_scaling":
               std = 1.0
               logging.info("No scaling selected")
            else:
                mean, std = modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
                )
            model = modules.ScaleShiftMACE(
            **model_config,
            correlation=correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
            )
        elif model == "ScaleShiftMACE":
            mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
            model = modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
            )
        elif model == "ScaleShiftBOTNet":
            mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
            model = modules.ScaleShiftBOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
            )
        elif model == "BOTNet":
            model = modules.BOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            )
        else:
            raise RuntimeError(f"Unknown model: '{model}'")

        model.to(device)

        # Optimizer
        decay_interactions = {}
        no_decay_interactions = {}
        for name, param in model.interactions.named_parameters():
            if "linear.weight" in name or "skip_tp_full.weight" in name:
                decay_interactions[name] = param
            else:
                no_decay_interactions[name] = param

        param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
            ],
            lr=args.lr,
            amsgrad=amsgrad,
        )

        optimizer: torch.optim.Optimizer
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(**param_options)
        else:
            optimizer = torch.optim.Adam(**param_options)

        logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

        if args.scheduler == "ExponentialLR":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=args.lr_scheduler_gamma
            )
        elif args.scheduler == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=args.lr_factor,
            patience=args.scheduler_patience,
            )
        else:
            raise RuntimeError(f"Unknown scheduler: '{args.scheduler}'")

        checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir, tag=tag, keep=args.keep_checkpoints
        )

        start_epoch = 0
        if restart_latest:
            opt_start_epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler), device=device
            )
            if opt_start_epoch is not None:
                start_epoch = opt_start_epoch

        swa: Optional[tools.SWAContainer] = None
        if swa:
            if start_swa is None:
                start_swa = (
                max_num_epochs // 4 * 3
                )  # if not set start swa at 75% of training
            if loss == "forces_only":
                logging.info("Can not select swa with forces only loss.")
            loss_fn_energy = modules.WeightedEnergyForcesLoss(
            energy_weight=args.swa_energy_weight, forces_weight=args.swa_forces_weight
            )
            swa = tools.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(
                optimizer=optimizer,
                swa_lr=args.swa_lr,
                anneal_epochs=1,
                anneal_strategy="linear",
            ),
            start=start_swa,
            loss_fn=loss_fn_energy,
            )
            logging.info(
            f"Using stochastic weight averaging (after {swa.start} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight} and learning rate : {args.swa_lr}"
            )

        ema: Optional[ExponentialMovingAverage] = None
        if ema:
            ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

        logging.info(model)
        logging.info(f"Number of parameters: {tools.count_parameters(model)}")
        logging.info(f"Optimizer: {optimizer}")

        tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs= max_num_epochs,
        logger=logger,
        patience=args.patience,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=error_table,
        )

        epoch = checkpoint_handler.load_latest(
        state=tools.CheckpointState(model, optimizer, lr_scheduler), device=device
        )
        logging.info(f"Loaded model from epoch {epoch}")

        # Evaluation on test datasets
        logging.info("Computing metrics for training, validation, and test sets")
        
        all_collections = [
        ("train", collections.train),
        ("valid", collections.valid),
        ]

        table = self.create_error_table(
        error_table,
        all_collections,
        z_table,
        r_max,
        args.valid_batch_size,
        model,
        loss_fn,
        device)
        # Save entire model
        model_path = os.path.join(args.checkpoints_dir, tag + ".model")

        logging.info("\n" + str(table))

        logging.info(f"Saving model to {model_path}")
        if save_cpu:
            model = model.to("cpu")
        torch.save(model, model_path)

        logging.info("Done")

        return None
    
    def _train(self, **kwargs):

        n_cores = kwargs['n_cores'] if 'n_cores' in kwargs else min(Config.n_cores, 8)
        os.environ['OMP_NUM_THREADS'] = str(n_cores)
        logger.info(f'Using {n_cores} cores to train MACE potential')

        for config in self.training_data:
            if self.requires_non_zero_box_size and config.box is None:
                config.box = Box([100, 100, 100])

        self.training_data.save_xyz(filename=f'{self.name}_data.xyz')
        start_time = time()
        self._train_file()
        delta_time = time() - start_time
        logger.info(f'MACE training ran in {delta_time / 60:.1f} m')

        return None

    def get_dataset_from_xyz(self,  train_path, valid_path, valid_fraction, config_type_weights, seed, energy_key, forces_key):
        """Load training and test dataset from xyz file"""
        
        try:
            from mace import data

        except ModuleNotFoundError:
            raise ModuleNotFoundError('MACE install not found, install it '
                                      'here: https://github.com/ACEsuit/mace')
            
        _, all_train_configs = data.load_from_xyz(file_path=train_path,
                                                  config_type_weights=config_type_weights,
                                                  energy_key=energy_key,
                                                  forces_key=forces_key,
                                                  extract_atomic_energies=False)
        logging.info(f"Loaded {len(all_train_configs)} training configurations from '{train_path}'")

        if valid_path is not None:
            _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False)
            logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'")
            train_configs = all_train_configs
        else:
            logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction)
            train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed)

        return (SubsetCollection(train=train_configs, valid=valid_configs))
    
    def create_error_table( self, table_type, all_collections, z_table, r_max, valid_batch_size, model, loss_fn, device):
        table = PrettyTable()
        
        try:
            from mace import data
            from mace.tools import torch_geometric,  evaluate

        except ModuleNotFoundError:
            raise ModuleNotFoundError('MACE install not found, install it '
                                      'here: https://github.com/ACEsuit/mace')
            
        if table_type == "TotalRMSE":
            table.field_names = [
            "config_type",
            "RMSE E / meV",
            "RMSE F / meV / A",
            "relative F RMSE %",
            ]
        elif table_type == "PerAtomRMSE":
            table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            ]
        elif table_type == "TotalMAE":
            table.field_names = [
            "config_type",
            "MAE E / meV",
            "MAE F / meV / A",
            "relative F MAE %",
            ]
        elif table_type == "PerAtomMAE":
            table.field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
            ]
        for name, subset in all_collections:
            data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
                for config in subset
            ],
            batch_size=valid_batch_size,
            shuffle=False,
            drop_last=False,
            )

            logging.info(f"Evaluating {name} ...")
            _, metrics = evaluate(
            model, loss_fn=loss_fn, data_loader=data_loader, device=device
            )
            if table_type == "TotalRMSE":
                table.add_row(
                [
                    name,
                    f"{metrics['rmse_e'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                ]
                )
            elif table_type == "PerAtomRMSE":
                table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:.1f}",
                    f"{metrics['rmse_f'] * 1000:.1f}",
                    f"{metrics['rel_rmse_f']:.2f}",
                ]
                )
            elif table_type == "TotalMAE":
                table.add_row(
                [
                    name,
                    f"{metrics['mae_e'] * 1000:.1f}",
                    f"{metrics['mae_f'] * 1000:.1f}",
                    f"{metrics['rel_mae_f']:.2f}",
                ]
                )
            elif table_type == "PerAtomMAE":
                table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:.1f}",
                    f"{metrics['mae_f'] * 1000:.1f}",
                    f"{metrics['rel_mae_f']:.2f}",
                ]
                )
        return table

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
            from mace.calculators import MACECalculator
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MACE install not found, install it '
                                      'here: https://github.com/ACEsuit/mace')
        
        calculator = MACECalculator(model_path = self.path+f"/checkpoints/{self.name}.model",
                                   device='cpu', default_dtype="float64")
        return calculator

    def copy(self):
        return deepcopy(self)
    
    @property
    def filename(self):
        return f'{self.name}.model'

    def __init__(self,
                 name,
                 system,
                 path) -> None:
        super().__init__(name = name, system = system)
        self.path = path
        
