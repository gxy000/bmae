# Copyright (c) Meta Platforms, Inc. and affiliates.
# DINO-specific training loop adapted from facebookresearch/dino.

import math
import sys
from typing import Iterable

import numpy as np
import torch

import util.misc as misc


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for name, param in model.named_parameters():
        if "head.last_layer" in name:
            param.grad = None


def _scaler_step(
    loss,
    optimizer,
    parameters,
    loss_scaler,
    clip_grad,
    update_grad,
    model,
    epoch,
    freeze_last_layer,
):
    loss_scaler._scaler.scale(loss).backward()
    if not update_grad:
        return None

    if clip_grad:
        loss_scaler._scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
    else:
        loss_scaler._scaler.unscale_(optimizer)
        grad_norm = misc.get_grad_norm_(parameters)

    cancel_gradients_last_layer(epoch, model, freeze_last_layer)
    loss_scaler._scaler.step(optimizer)
    loss_scaler._scaler.update()
    return grad_norm


def train_one_epoch(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dino_loss: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    log_writer=None,
    args=None,
):
    student.train(True)
    teacher.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("wd", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("mom", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    student_without_ddp = student.module if hasattr(student, "module") else student
    parameters = [p for p in student_without_ddp.parameters() if p.requires_grad]

    for data_iter_step, crops in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = len(data_loader) * epoch + data_iter_step
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[it]
            if param_group.get("weight_decay", 0) > 0:
                param_group["weight_decay"] = wd_schedule[it]

        if isinstance(crops, torch.Tensor):
            crops = [crops]
        crops = [crop.to(device, non_blocking=True) for crop in crops]

        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            with torch.no_grad():
                teacher_output = teacher(crops[:2])
            student_output = student(crops)
            loss = dino_loss(student_output, teacher_output, epoch)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        update_grad = (data_iter_step + 1) % accum_iter == 0
        grad_norm = _scaler_step(
            loss / accum_iter,
            optimizer,
            parameters,
            loss_scaler,
            args.clip_grad,
            update_grad,
            student,
            epoch,
            args.freeze_last_layer,
        )

        if update_grad:
            optimizer.zero_grad()
            with torch.no_grad():
                momentum = momentum_schedule[it]
                for param_q, param_k in zip(student_without_ddp.parameters(), teacher.parameters()):
                    param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
        else:
            momentum = momentum_schedule[it]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0].get("weight_decay", 0))
        metric_logger.update(mom=momentum)
        if grad_norm is not None:
            metric_logger.update(grad_norm=float(grad_norm))

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and update_grad:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_1000x)
            log_writer.add_scalar(
                "wd", optimizer.param_groups[0].get("weight_decay", 0), epoch_1000x
            )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
