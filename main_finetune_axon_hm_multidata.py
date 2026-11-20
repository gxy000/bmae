# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
import timm.optim.optim_factory as optim_factory

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


import models_mae_axon_hm_multidata as models_mae_axon_hm
from models_mae_axon_hm_multidata import SetCriterion
from engine_finetune_axon_hm_multidata import train_one_epoch, evaluate
from util.axon_seg_datasets_hm_multidata import AxonSegDataset, collate_detr

# from util.axon_seg_datasets_hm_junction_multidata import AxonSegDataset, collate_detr

torch.manual_seed(42)


def get_args_parser():
    parser = argparse.ArgumentParser("Neurite heatmap/instance finetuning", add_help=True)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_base_patch16_1c_32",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)"
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay", type=float, default=0.75, help="layer-wise lr decay from ELECTRA/BEiT"
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    # Axon Seg Loss function parameters
    parser.add_argument("--weight_hm", default=1.0, type=float)
    parser.add_argument("--weight_de", default=1.0, type=float)
    parser.add_argument("--threshold_pos", default=1.0, type=float)
    parser.add_argument("--delta_var", default=0.5, type=float)
    parser.add_argument("--delta_dist", default=1.5, type=float)
    parser.add_argument("--de_alpha", default=1.0, type=float)
    parser.add_argument("--de_beta", default=1.0, type=float)
    parser.add_argument("--de_gamma", default=0.001, type=float)
    parser.add_argument("--max_samples", default=4096, type=int)
    parser.add_argument("--max_per_inst", default=1024, type=int)
    parser.add_argument("--min_vox_per_inst", default=4, type=int)
    parser.add_argument(
        "--weighted_disc_loss",
        action="store_true",
        help="Set to True to weight disc loss by instance size",
    )  # axon seg specific parameter

    # Dataset parameters
    parser.add_argument(
        "--data_paths", nargs="+", default=[], type=str, help="list of dataset paths"
    )
    parser.add_argument(
        "--train_datas",
        nargs="+",
        default=[],
        type=str,
        help="list of json files for training data",
    )
    parser.add_argument(
        "--mixture_weights",
        nargs="+",
        default=[],
        type=float,
        help="list of sampling weights for each dataset",
    )
    parser.add_argument(
        "--nb_classes", default=1000, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--marker_only", action="store_true", help="Set to True for 1 channel detection"
    )
    # parser.add_argument('--cell_size', default=16, type=int, help='average cell diameter') # cell detection specific parameter, not used
    parser.add_argument(
        "--point2segment",
        action="store_true",
        help="Set to True to convert points to segments for axon seg training",
    )  # axon seg specific parameter
    parser.add_argument(
        "--max_vol_num",
        default=5,
        type=int,
        help="max num of volume to overlay together to create synthesized data",
    )
    parser.add_argument(
        "--subvol_size", nargs=3, default=(32, 256, 256), type=int, help="input volume to model"
    )
    parser.add_argument("--dataset_size", default=512, type=int, help="num of data in each epoch")
    parser.add_argument(
        "--norm_values",
        nargs=2,
        default=None,
        type=float,
        help="min and max values for normalization",
    )

    parser.add_argument(
        "--output_dir",
        default="/media/hddshare/general/mae_brain/brain-mae/output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="/media/hddshare/general/mae_brain/brain-mae/output_dir",
        help="path where to tensorboard log",
    )
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--output_patch_size",
        type=int,
        default=2,
        help="Output stride for hm prediction downsample",
    )
    parser.add_argument(
        "--transforms",
        default="",
        type=str,
        help="augmentations: c-center_crop, r-resize, f-flip, g-gamma",
    )

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str,
        help="Weights & Biases project name; enables logging when set.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        type=str,
        help="Optional Weights & Biases entity to log under.",
    )
    parser.add_argument(
        "--wandb_run_name", default=None, type=str, help="Optional Weights & Biases run name."
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    # torch.cuda.set_device(0)

    wandb_run = None
    if wandb is not None and misc.is_main_process() and args.wandb_project:
        init_kwargs = dict(project=args.wandb_project, config=vars(args))
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        if args.output_dir:
            init_kwargs.setdefault("dir", args.output_dir)
        try:
            wandb_run = wandb.init(**init_kwargs)
        except Exception as exc:
            print(f"Failed to initialize Weights & Biases logging: {exc}")
            wandb_run = None

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # transforms = ["resize", "flip", "gamma"] # augmentation in major experiments from 6/19
    # transforms = ["flip", "gamma"] # major experiments
    transforms = []
    if "c" in args.transforms:
        transforms.append("center_crop")
    else:
        transforms.append("random_crop")
    if "r" in args.transforms:
        transforms.append("resize")
    if "f" in args.transforms:
        transforms.append("flip")
    if "g" in args.transforms:
        transforms.append("gamma")
    if "d" in args.transforms:
        transforms.append("rotate")

    params = {
        "resize_ratio": (0.8, 1.2),
        "resize_mode": "trilinear",
        "rotate_degrees": (0, 45),
        "gamma_range": (0.8, 2),
        "flip_axes": [-1, -2, -3],
        "norm_percentiles": (1, 99),
    }
    if args.norm_values is not None:
        print("Using normalization by values:", args.norm_values)
        transforms.append("normalize_by_value")
        params["norm_values"] = tuple(args.norm_values)
    print(f"training with transform: {transforms}")

    if "1c" in args.model:
        twochannel = False
    else:
        twochannel = True

    dataset_train = AxonSegDataset(
        data_paths=args.data_paths,
        json_paths=args.train_datas,
        point2segment=args.point2segment,
        mixture_weights=args.mixture_weights,
        dataset_size=args.dataset_size,
        max_vol_num=args.max_vol_num,
        subvol_size=args.subvol_size,
        R=args.output_patch_size,
        twochannel=twochannel,
        transforms=transforms,
        params=params,
    )
    dataset_val = AxonSegDataset(
        data_paths=args.data_paths,
        json_paths=args.train_datas,
        point2segment=args.point2segment,
        mixture_weights=args.mixture_weights,
        dataset_size=16,
        max_vol_num=1,
        subvol_size=args.subvol_size,
        R=args.output_patch_size,
        twochannel=twochannel,
        transforms=transforms,
        params=params,
    )

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
            # persistent_workers=False,
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
                # persistent_workers=False,
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_detr,
        persistent_workers=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,  # args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_detr,
        persistent_workers=False,
    )

    model = models_mae_axon_hm.__dict__[args.model]()

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    # linear prob
    # for _, p in model.named_parameters():
    #     p.requires_grad = False
    # for _, p in model.rpn.named_parameters():
    #     p.requires_grad = True
    # for _, p in model.named_parameters():
    #     p.requires_grad = False
    # for _, p in model.CenterNet.named_parameters():
    #     p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ------------------------------ Criterion ------------------------------
    weight_dict = {"loss_hm": args.weight_hm, "loss_de": args.weight_de}
    criterion = SetCriterion(
        weight_dict=weight_dict,
        threshold_pos=args.threshold_pos,
        delta_var=args.delta_var,
        delta_dist=args.delta_dist,
        alpha=args.de_alpha,
        beta=args.de_beta,
        gamma=args.de_gamma,
        max_samples=args.max_samples,
        max_per_inst=args.max_per_inst,
        min_vox_per_inst=args.min_vox_per_inst,
        weighted_disc_loss=args.weighted_disc_loss,
    ).to(device)

    # ------------------------------ Optimizer ------------------------------
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.8, 0.95), lr=args.lr)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler
    )

    # tensorboard
    log_writer = None
    if global_rank == 0 and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(args.log_dir)

    if args.eval:
        test_stats = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation stats on {len(dataset_val)} samples: {test_stats}")
        if wandb_run is not None:
            wandb.log({f"val_{k}": v for k, v in test_stats.items()})
            wandb_run.finish()
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and epoch % 50 == 49:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        misc.save_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            epoch_name="max",
        )

        val_stats = evaluate(model, criterion, data_loader_val, device)

        # log
        log_stats = {f"train_{k}": v for k, v in train_stats.items()}
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
        log_stats["epoch"] = epoch

        if wandb_run is not None:
            wandb.log(log_stats, step=epoch)

        if misc.is_main_process() and args.output_dir:
            with open(Path(args.output_dir) / "log.txt", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            if log_writer:
                for k, v in train_stats.items():
                    log_writer.add_scalar(k, v, epoch)
                for k, v in val_stats.items():
                    log_writer.add_scalar("val_" + k, v, epoch)
                log_writer.flush()

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
