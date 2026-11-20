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

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
import timm.optim.optim_factory as optim_factory

assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# import models_mae_rpn
# import models_mae_cg
import models_mae_cn_hm_offset

# rpn
# from engine_finetune import train_one_epoch, evaluate
# from util.cell_detection_datasets import CellDetectionDatset
# cgn
# from engine_finetune_cg import train_one_epoch, evaluate
# from util.cell_detection_datasets_cg import CellDetectionDatset
# centernet
from engine_finetune_cn_hm_offset import train_one_epoch, evaluate
from util.cell_detection_datasets_cn_hm_offset import CellDetectionDatset

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# torch.manual_seed(42)


def get_args_parser():
    parser = argparse.ArgumentParser("Cell heatmap/offset finetuning", add_help=True)
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
        default="mae_vit_base_patch16_4",
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

    parser.add_argument("--rpn_neg_threshold", default=0.3, type=float)
    parser.add_argument("--rpn_pos_threshold", default=0.7, type=float)
    parser.add_argument("--dist_weight", default=0.1, type=float)
    parser.add_argument("--background_loss_weight", default=0.2, type=float)
    parser.add_argument("--focalloss_pos", default=1, type=float)

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/media/hddshare/general/mae_brain/data/bmae",
        type=str,
        help="dataset path",
    )
    parser.add_argument("--train_data", default="data/EGFP_200_3x512.json", type=str)
    parser.add_argument(
        "--nb_classes", default=1000, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--marker_only", action="store_true", help="Set to True for 1 channel detection"
    )
    parser.add_argument("--cell_size", default=16, type=int, help="average cell diameter")

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
        "--R", type=int, default=2, help="Output stride for hm prediction downsample"
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
    print(f"training with transform: {transforms}")
    # rpn
    # dataset_train = CellDetectionDatset(data_path=args.data_path, json_path=args.train_data, anchors_per_image=512, negative_thre=args.rpn_neg_threshold, positive_thre=args.rpn_pos_threshold, transforms=transforms, params=params)
    # dataset_val = CellDetectionDatset(data_path=args.data_path, json_path='data/train.json', anchors_per_image=512, negative_thre=args.rpn_neg_threshold, positive_thre=args.rpn_pos_threshold, transforms=transforms, params=params)

    # cgn
    # dataset_train = CellDetectionDatset(data_path=args.data_path, json_path=args.train_data,
    #                                     marker_only=args.marker_only,
    #                                     transforms=transforms, params=params,
    #                                     box_width=args.cell_size)
    # dataset_val = CellDetectionDatset(data_path=args.data_path, json_path='data/train.json',
    #                                   marker_only=args.marker_only,
    #                                   transforms=transforms, params=params,
    #                                   batched=False,
    #                                   box_width=args.cell_size)

    # cn
    two_channel = False if "1c" in args.model else True
    dataset_train = CellDetectionDatset(
        data_path=args.data_path,
        json_path=args.train_data,
        marker_only=args.marker_only,
        two_channel=two_channel,
        transforms=transforms,
        params=params,
        box_width=args.cell_size,
        R=args.R,
    )
    dataset_val = CellDetectionDatset(
        data_path=args.data_path,
        json_path="data/EGFP_200_val.json",
        marker_only=args.marker_only,
        two_channel=two_channel,
        transforms=transforms,
        params=params,
        batched=False,
        box_width=args.cell_size,
        R=args.R,
    )

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
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
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
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
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,  # args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # model = models_mae_rpn.__dict__[args.model]()
    # model = models_mae_cg.__dict__[args.model]()
    model = models_mae_cn_hm_offset.__dict__[args.model](output_patch_size=args.R)

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

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler
    )

    if args.eval:
        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            args.dist_weight,
            args.background_loss_weight,
            args.focalloss_pos,
        )
        print(f"Validation loss on {len(dataset_val)} volumes: {test_stats['loss']:.6f}")
        if wandb_run is not None:
            wandb.log({f"test_{k}": v for k, v in test_stats.items()})
            wandb_run.finish()
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
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

        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            args.dist_weight,
            args.background_loss_weight,
            args.focalloss_pos,
        )
        # print(f"Validation loss on {len(dataset_val)} volumes: {test_stats['loss']:.6f}")
        # max_accuracy = max(max_accuracy, test_stats["acc1"])
        # print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            # log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
            log_writer.add_scalar("perf/test_loss_pred_hm", test_stats["loss_pred_hm"], epoch)
            log_writer.add_scalar("perf/test_loss_pred_off", test_stats["loss_pred_off"], epoch)

            # log_writer.add_scalar('perf/test_loss_dist', test_stats['loss_dist'], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if wandb_run is not None:
            wandb.log(log_stats, step=epoch)

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
