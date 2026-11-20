# Copyright (c) Meta Platforms, Inc. and affiliates.
# BrainDINO pretraining entry point.

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import models_dino
import util.misc as misc
from engine_pretrain_DINO import cosine_scheduler, train_one_epoch
from util.DINO_datasets import BrainDINOMultiCropDataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

try:
    import wandb

except ImportError:
    wandb = None


def bool_flag(s):
    if isinstance(s, bool):
        return s
    falsy = {"off", "false", "0"}
    truthy = {"on", "true", "1"}
    if s.lower() in falsy:
        return False
    if s.lower() in truthy:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def get_args_parser():
    parser = argparse.ArgumentParser("BrainDINO pre-training", add_help=True)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU.")
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    parser.add_argument(
        "--model",
        default="mae_vit_base_patch16",
        type=str,
        help="DINO model name. MAE aliases are supported for checkpoint compatibility.",
    )
    parser.add_argument(
        "--out_dim", default=65536, type=int, help="DINO head output dimensionality."
    )
    parser.add_argument("--norm_last_layer", default=True, type=bool_flag)
    parser.add_argument("--use_bn_in_head", default=False, type=bool_flag)
    parser.add_argument("--head_nlayers", default=3, type=int)
    parser.add_argument("--head_hidden_dim", default=2048, type=int)
    parser.add_argument("--head_bottleneck_dim", default=256, type=int)

    parser.add_argument("--global_crop_size", default=96, type=int)
    parser.add_argument("--local_crop_size", default=64, type=int)
    parser.add_argument("--parent_size", default=128, type=int)
    parser.add_argument("--local_crops_number", default=8, type=int)
    parser.add_argument(
        "--transforms",
        default="rfg",
        type=str,
        help="DINO view augmentations: r-resize, f-flip, g-gamma.",
    )
    parser.add_argument("--resize_ratio", default=(0.8, 1.2), type=float, nargs=2)
    parser.add_argument("--gamma_range", default=(0.8, 2.0), type=float, nargs=2)

    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.05,
        help="Final cosine weight decay. Use 0.4 for the original DINO ramp.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--blr",
        type=float,
        default=1.5e-4,
        help="Base LR: actual_lr = blr * total_batch_size / 256.",
    )
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--opt_betas", default=(0.9, 0.95), type=float, nargs=2)

    parser.add_argument("--momentum_teacher", default=0.996, type=float)
    parser.add_argument("--warmup_teacher_temp", default=0.04, type=float)
    parser.add_argument("--teacher_temp", default=0.04, type=float)
    parser.add_argument("--warmup_teacher_temp_epochs", default=0, type=int)
    parser.add_argument("--student_temp", default=0.1, type=float)
    parser.add_argument("--center_momentum", default=0.9, type=float)
    parser.add_argument("--freeze_last_layer", default=1, type=int)
    parser.add_argument("--use_fp16", default=True, type=bool_flag)

    parser.add_argument(
        "--data_path",
        default="/media/hddshare/general/mae_brain/data/bmae/20240418_11_27_21_Xinyi_Thy1_EGFP_TH_Syto",
        type=str,
    )
    parser.add_argument("--nuclei_folder", default="Ex_642_Em_2_precomputed", type=str)
    parser.add_argument("--marker_folder", default="Ex_488_Em_0_precomputed", type=str)
    parser.add_argument("--num_of_data", default=100000, type=int)
    parser.add_argument("--min_intensity", default=150, type=float)

    parser.add_argument(
        "--output_dir", default="/media/hddshare/general/mae_brain/brain-mae/output_dir"
    )
    parser.add_argument("--export_name", default="brainDINO_encoder.pth")
    parser.add_argument("--save_freq", default=50, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)

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

    parser.add_argument(
        "--verify_celldet",
        action="store_true",
        default=True,
        help="Verify exported encoder keys load into BMAE-CellDet.",
    )
    parser.add_argument("--no_verify_celldet", action="store_false", dest="verify_celldet")
    parser.add_argument(
        "--celldet_model",
        default=None,
        type=str,
        help="CellDet model used for export verification.",
    )
    parser.add_argument("--celldet_output_patch_size", default=2, type=int)
    parser.add_argument(
        "--skip_celldet_forward",
        action="store_true",
        help="Only verify names/shapes/load_state_dict, skip dummy forward.",
    )

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def build_transform_list(transform_codes):
    transforms = []
    if "r" in transform_codes:
        transforms.append("resize")
    if "f" in transform_codes:
        transforms.append("flip")
    if "g" in transform_codes:
        transforms.append("gamma")
    return transforms


def get_model_without_ddp(model):
    return model.module if hasattr(model, "module") else model


def save_dino_checkpoint(args, epoch, student, teacher, optimizer, loss_scaler, dino_loss):
    if not args.output_dir:
        return
    output_dir = Path(args.output_dir)
    student_without_ddp = get_model_without_ddp(student)
    to_save = {
        "student": student_without_ddp.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "dino_loss": dino_loss.state_dict(),
        "args": args,
    }
    checkpoint_paths = [output_dir / "checkpoint.pth", output_dir / ("checkpoint-%s.pth" % epoch)]
    for checkpoint_path in checkpoint_paths:
        misc.save_on_master(to_save, checkpoint_path)


def export_encoder_checkpoint(args, epoch, student):
    student_without_ddp = get_model_without_ddp(student)
    encoder_state = student_without_ddp.encoder_state_dict()
    export = {
        "model": encoder_state,
        "epoch": epoch,
        "args": args,
        "pretrain_method": "BrainDINO",
        "note": "Contains only BrainMAE/CellDet-compatible student encoder parameters.",
    }
    path = Path(args.output_dir) / args.export_name
    misc.save_on_master(export, path)
    return path, encoder_state


def infer_celldet_model_name(args, in_chans):
    if args.celldet_model:
        return args.celldet_model
    if in_chans == 1:
        return "mae_vit_base_patch16_4_1c"
    return "mae_vit_base_patch16_4"


def verify_celldet_compatibility(args, encoder_state, device):
    import models_mae_cn_hm_offset as models_celldet

    in_chans = encoder_state["patch_embed.proj.weight"].shape[1]
    celldet_model_name = infer_celldet_model_name(args, in_chans)
    celldet = models_celldet.__dict__[celldet_model_name](
        output_patch_size=args.celldet_output_patch_size
    )
    target_state = celldet.state_dict()
    encoder_keys = [
        k
        for k in target_state
        if k == "cls_token"
        or k == "pos_embed"
        or k.startswith(("patch_embed.", "blocks.", "norm."))
    ]
    missing_from_export = [k for k in encoder_keys if k not in encoder_state]
    unexpected = [k for k in encoder_state if k not in target_state]
    shape_mismatch = [
        (k, tuple(encoder_state[k].shape), tuple(target_state[k].shape))
        for k in encoder_keys
        if k in encoder_state and tuple(encoder_state[k].shape) != tuple(target_state[k].shape)
    ]
    if missing_from_export or unexpected or shape_mismatch:
        raise RuntimeError(
            "BrainDINO encoder is not CellDet-compatible: "
            f"missing={missing_from_export}, unexpected={unexpected}, shape_mismatch={shape_mismatch}"
        )

    msg = celldet.load_state_dict(encoder_state, strict=False)
    if len(msg.unexpected_keys) != 0:
        raise RuntimeError(f"Unexpected keys when loading exported encoder into CellDet: {msg}")

    if not args.skip_celldet_forward:
        celldet.to(device)
        celldet.eval()
        sample = torch.zeros(1, in_chans, 128, 128, 128, device=device)
        with torch.no_grad():
            _ = celldet(sample)
        celldet.cpu()

    print(
        "Verified BrainDINO encoder export against "
        f"{celldet_model_name}: {len(encoder_keys)} encoder tensors loaded; "
        f"missing downstream-only tensors={len(msg.missing_keys)}."
    )


def load_dino_checkpoint(args, student, teacher, optimizer, loss_scaler, dino_loss):
    if not args.resume:
        return
    checkpoint = torch.load(args.resume, map_location="cpu")
    student_without_ddp = get_model_without_ddp(student)
    if "student" in checkpoint:
        student_without_ddp.load_state_dict(checkpoint["student"])
    elif "model" in checkpoint:
        msg = student_without_ddp.backbone.load_state_dict(checkpoint["model"], strict=False)
        print(f"Loaded encoder-only checkpoint into student backbone: {msg}")
    else:
        student_without_ddp.load_state_dict(checkpoint)

    if "teacher" in checkpoint:
        teacher.load_state_dict(checkpoint["teacher"])
    else:
        teacher.load_state_dict(student_without_ddp.state_dict())
    if "optimizer" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
    if "scaler" in checkpoint:
        loss_scaler.load_state_dict(checkpoint["scaler"])
    if "dino_loss" in checkpoint:
        dino_loss.load_state_dict(checkpoint["dino_loss"])
    print("Resume checkpoint %s" % args.resume)


def main(args):
    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

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

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    transforms = build_transform_list(args.transforms)
    params = {
        "resize_ratio": tuple(args.resize_ratio),
        "resize_mode": "trilinear",
        "gamma_range": tuple(args.gamma_range),
        "flip_axes": [-1, -2, -3],
        "norm_percentiles": (1, 99),
    }
    dataset_train = BrainDINOMultiCropDataset(
        data_path=args.data_path,
        data_folder_list=[args.nuclei_folder, args.marker_folder],
        num_of_data=args.num_of_data,
        parent_size=(args.parent_size, args.parent_size, args.parent_size),
        global_size=(args.global_crop_size, args.global_crop_size, args.global_crop_size),
        local_size=(args.local_crop_size, args.local_crop_size, args.local_crop_size),
        local_crops_number=args.local_crops_number,
        transforms=transforms,
        params=params,
        min_intensity=args.min_intensity,
    )
    print(dataset_train)
    print(f"DINO view transforms: {transforms}, params: {params}")

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    student = models_dino.__dict__[args.model](
        out_dim=args.out_dim,
        use_bn_in_head=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        head_nlayers=args.head_nlayers,
        head_hidden_dim=args.head_hidden_dim,
        head_bottleneck_dim=args.head_bottleneck_dim,
    )
    teacher = models_dino.__dict__[args.model](
        out_dim=args.out_dim,
        use_bn_in_head=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        head_nlayers=args.head_nlayers,
        head_hidden_dim=args.head_hidden_dim,
        head_bottleneck_dim=args.head_bottleneck_dim,
    )
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    print("Student = %s" % str(student))

    if args.verify_celldet and misc.is_main_process():
        verify_celldet_compatibility(args, student.encoder_state_dict(), device)

    student.to(device)
    teacher.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.gpu], find_unused_parameters=True
        )

    student_without_ddp = get_model_without_ddp(student)
    param_groups = optim_factory.add_weight_decay(student_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=tuple(args.opt_betas))
    print(optimizer)
    loss_scaler = NativeScaler()

    dino_loss = models_dino.DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    lr_schedule = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, len(data_loader_train), warmup_epochs=args.warmup_epochs
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader_train)
    )
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader_train)
    )

    load_dino_checkpoint(args, student, teacher, optimizer, loss_scaler, dino_loss)

    print(f"Start BrainDINO training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student,
            teacher,
            dino_loss,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            save_dino_checkpoint(args, epoch, student, teacher, optimizer, loss_scaler, dino_loss)
            export_path, encoder_state = export_encoder_checkpoint(args, epoch, student)
            if args.verify_celldet and misc.is_main_process():
                print(f"Exported CellDet-compatible encoder to {export_path}")
                verify_celldet_compatibility(args, encoder_state, device)

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
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
