# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch


import util.misc as misc
import util.lr_sched as lr_sched

# from util.cell_detection_datasets_cg import compute_centers

# from util.nms import nms


def _neg_loss(pred, gt, threshold_pos, alpha=0.5):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    """

    pos_inds = gt.ge(threshold_pos).float()
    neg_inds = gt.lt(threshold_pos).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    # pos_loss = alpha*pos_loss.sum()
    # neg_loss = (1-alpha)*neg_loss.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / (num_pos + 1e-6)

    return loss


def FocalLoss(out, target, threshold_pos=1):
    """nn.Module warpper for focal loss"""
    return _neg_loss(out, target, threshold_pos, alpha=0.9)


def weighted_l1(pred, target, weight):
    # l1 loss
    mask = (target > 0.9).int()
    # TODO background_loss_weight
    weighted_loss = torch.sum(torch.abs(pred - target) * mask) / (
        torch.sum(mask) + 1e-6
    ) + weight * torch.sum(torch.abs(pred - target) * (1 - mask)) / (torch.sum(1 - mask) + 1e-6)
    return weighted_loss


def compute_off_loss(hm_mask, pred_off, target_off):
    # l1 loss
    pred = pred_off * torch.stack((hm_mask, hm_mask, hm_mask), dim=1)
    target = target_off * torch.stack((hm_mask, hm_mask, hm_mask), dim=1)
    # off_loss = torch.nn.L1Loss(pred, target, reduction='sum')/(torch.sum(hm_mask) + 1e-6)
    # l1 loss
    off_loss = torch.sum(torch.abs(pred - target)) / (torch.sum(hm_mask) + 1e-6)
    return off_loss


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (voxel, gt_hm, gt_off) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        voxel = voxel.to(device, non_blocking=True)
        gt_hm = gt_hm.to(device, non_blocking=True)
        gt_off = gt_off.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            pred_hm, pred_off = model(voxel)
            # pred_hm = model(voxel)

            loss_hm = FocalLoss(pred_hm, gt_hm, args.focalloss_pos)
            loss_off = compute_off_loss(
                ((gt_hm > args.focalloss_pos).sum(dim=1) > 0).int(), pred_off.unsqueeze(1), gt_off
            )
            loss = loss_hm + args.dist_weight * loss_off

        loss_value = loss.item()
        loss_pred_hm_value = loss_hm.item()
        loss_pred_off_value = loss_off.item()
        # loss_dist_value = loss_dist.item()

        if not math.isfinite(loss_value):
            print(f"loss break down: gradient {loss_pred_hm_value}")
            print("Loss is {}, stopping training".format(loss_value))
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
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_pred_hm=loss_pred_hm_value)
        metric_logger.update(loss_pred_off=loss_pred_off_value)

        # metric_logger.update(loss_dist=loss_dist_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("loss_pred_hm", loss_pred_hm_value, epoch_1000x)
            log_writer.add_scalar("loss_pred_off", loss_pred_off_value, epoch_1000x)
            # log_writer.add_scalar('loss_dist', loss_dist_value, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, dist_weight, background_loss_weight, focalloss_pos):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 100, header):
        voxel, gt_hm, gt_off, centers = batch
        voxel = voxel.to(device, non_blocking=True)
        gt_hm = gt_hm.to(device, non_blocking=True)
        gt_off = gt_off.to(device, non_blocking=True)

        # compute output
        with (torch.cuda.amp.autocast()):
            pred_hm, pred_off = model(voxel)

            loss_pred_hm = FocalLoss(pred_hm, gt_hm, threshold_pos=focalloss_pos)
            loss_pred_off = compute_off_loss(
                ((gt_hm > focalloss_pos).sum(dim=1) > 0).int(), pred_off.unsqueeze(1), gt_off
            )
            loss = loss_pred_hm + dist_weight * loss_pred_off

        loss_value = loss.item()
        loss_pred_hm_value = loss_pred_hm.item()
        loss_pred_off_value = loss_pred_off.item()

        # if loss < 5:
        #     pred_grad = pred_grad[0].detach().cpu().numpy()
        #     pred_dist = pred_dist[0].detach().cpu().numpy()
        #     centers = centers[0].detach().cpu().numpy()
        #     pred_centers = compute_centers(pred_grad, pred_dist, niter=16, flow_threshold=0.8, max_size=8)
        #     tps, fns, fps = match_cell(pred_centers, centers, R=5)
        #     precision, recall, acc = evaluate_metrics(len(tps), len(fns), len(fps))
        # else:

        # acc1, acc2 = accuracy(rpn_class_logits, rpn_class_targets, topk=(1, 2))

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_pred_hm=loss_pred_hm_value)
        metric_logger.update(loss_pred_off=loss_pred_off_value)
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("* Loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))
    # print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
