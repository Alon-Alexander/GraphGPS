import datetime
import os
import torch
import logging

import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from graphgps.logger import create_logger

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

# Define hardwired checkpoint path
CHECKPOINT_PATH = './results/composition-zinc-GPS+RWSE/0/ckpt/599.ckpt'
#/home/yandex/MLWG2024/roeibenzion/GraphGPS/results/composition-zinc-GPS+RWSE/0/ckpt
def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None):
    checkpoint_path = './results/composition-zinc-GPS+RWSE/0/ckpt/599.ckpt'
    checkpoint = torch.load(checkpoint_path)

    # Print all keys in the checkpoint to understand its structure
    print("Checkpoint keys:", checkpoint.keys())
    """Load model, optimizer, and scheduler states from checkpoint."""
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    epoch = checkpoint.get('epoch', 599)
    return epoch

def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg."""
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)

def run_loop_settings():
    """Create main loop execution settings based on the current cfg."""
    if len(cfg.run_multiple_splits) == 0:
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        if args.repeat != 1:
            raise NotImplementedError("Multiple repeats of multiple splits is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)

def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)

    # Set PyTorch environment
    torch.set_num_threads(cfg.num_threads)

    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()

        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")

        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()

        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )

        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

        # Resume training from checkpoint if it exists
        start_epoch = 0
        if os.path.exists(CHECKPOINT_PATH):
            logging.info(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
            start_epoch = load_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler)
        else:
            logging.warning(f"No checkpoint found at {CHECKPOINT_PATH}. Starting from scratch.")

        # Print model info
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Training {name}")

        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")

    # Mark configuration as done if in batch mode
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
