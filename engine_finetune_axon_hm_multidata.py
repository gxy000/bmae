from typing import Iterable
import math
import sys

import torch
from torch import nn

import util.misc as misc
import util.lr_sched as lr_sched

# -----------------------------------------------------------------------------
# Train‑one‑epoch
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0.0,
    log_writer=None,
    args=None,
):
    """Single training epoch for maeViT + DETR."""

    model.train()
    criterion.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    accum_iter = args.accum_iter
    optimizer.zero_grad(set_to_none=True)

    for data_iter_step, (voxels, targets, dataset_idx) in enumerate(
        metric_logger.log_every(data_loader, 20, header)
    ):
        # dynamic LR per iteration
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )
        voxels = voxels.to(device, non_blocking=True)
        dataset_idx = dataset_idx.squeeze(1).to(device, non_blocking=True)
        # targets is list[dict] with dense heatmap/instance_map on CPU – move to GPU
        for t in targets:
            for k in ("heatmap", "instance_map"):
                if k in t:
                    t[k] = t[k].to(device, non_blocking=True)

        # forward
        outputs = model(voxels, dataset_idx)
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss diverged: ", loss_dict)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        # --- logging ---
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

        lr_current = max(g["lr"] for g in optimizer.param_groups)
        metric_logger.update(lr=lr_current)

        # TensorBoard
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_x1000 = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("lr", lr_current, epoch_x1000)
            for k, v in loss_dict.items():
                log_writer.add_scalar(k, v.item(), epoch_x1000)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# -----------------------------------------------------------------------------
# Evaluation (validation) loop
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module, criterion: nn.Module, data_loader, device: torch.device,
):
    model.eval()
    criterion.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Val:"

    for voxels, targets, dataset_idx in metric_logger.log_every(data_loader, 50, header):
        voxels = voxels.to(device, non_blocking=True)
        # dataset_idx = dataset_idx.to(device, non_blocking=True)
        dataset_idx = dataset_idx.squeeze(1).to(device, non_blocking=True)
        for t in targets:
            for k in ("heatmap", "instance_map"):
                if k in t:
                    t[k] = t[k].to(device, non_blocking=True)

        outputs = model(voxels, dataset_idx)
        loss_dict = criterion(outputs, targets)
        loss_value = sum(loss_dict.values()).item()

        metric_logger.update(loss=loss_value)
        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item()})

    metric_logger.synchronize_between_processes()
    print("* Validation loss {loss.global_avg:.3f}".format(loss=metric_logger.loss))
    return {k: m.global_avg for k, m in metric_logger.meters.items()}
