import argparse
import contextlib
import json
import sys
import torch
import numpy as np
import time
from util.cell_detection_datasets_cn_hm_offset import CellDetectionDatset, _cn_decode_one_cls
from util.bmae_utils import match_cell, evaluate_metrics
from util.visualizer import Visualizer
import models_mae_cn_hm_offset
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline
from pathlib import Path
from collections import defaultdict
import imageio

transforms = ["whole_volume"]
params = {"norm_percentiles": (1, 99)}


def get_args_parser():
    parser = argparse.ArgumentParser("Evaluate and plot")

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/hddshare/general/mae_brain/data/bmae",
        help="path to the dataset",
    )
    parser.add_argument(
        "--testdata",
        type=str,
        default="EGFP_200_eval4_0915",
        help="filename of test data in data folder",
    )
    parser.add_argument(
        "--downsample_ratio", type=int, default=2, help="Downsample ratio for gt heatmap and offset"
    )
    parser.add_argument(
        "--threshold_hm",
        nargs="+",
        type=float,
        default=[0.1],
        help="Minimum probability as center in heatmap. Pass one value for all models, or one per model.",
    )

    # Model parameters:
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="/media/hddshare/general/mae_brain/brain-mae/output_dir",
        help="root containing the checkpoint experiment folders",
    )
    parser.add_argument(
        "--eval_root",
        default="/media/hddshare/general/mae_brain/brain-mae/eval/celldet",
        help="root for evaluation metrics, logs and visualizations",
    )
    parser.add_argument(
        "--ckpt_folder",
        type=str,
        default="detetion",
        help="shared checkpoint directory used when --ckpt_folders is empty",
    )
    parser.add_argument(
        "--ckpt_folders",
        nargs="+",
        type=str,
        default=[],
        help="list of checkpoint directories, same length as models",
    )
    parser.add_argument(
        "--encoder", type=str, default="vitb", help="config of pretrained encoder, default: vitb"
    )
    parser.add_argument(
        "--model_archs", nargs="+", type=str, help="list of the model architectures to evaluate"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="list of the model directories, same length as model_archs",
    )
    parser.add_argument(
        "--plot_labels",
        nargs="+",
        type=str,
        help="list of the model name shown in plot, same length as models",
    )
    parser.add_argument(
        "--pred_heads",
        type=str,
        default="hm_off",
        help="list of the prediction heads, default: hm_off",
    )
    parser.add_argument("--input_size", type=int, default=128, help="input size of the model")

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        default=[],
        help="checkpoint of each model, same length as model_archs and model_name, if empty, then checkpoint-199.pth by default",
    )

    # Inference parameters

    # Plot parameters
    parser.add_argument(
        "--R",
        nargs="+",
        type=int,
        default=[10],
        help="range of distance between gt to prediction for true positive, can be an integer or an int range. R and dist cant be range at the same time. closed interval",
    )
    parser.add_argument(
        "--dist", nargs="+", type=int, default=[2], help="Minimum distance between cells"
    )
    parser.add_argument(
        "--plot_R",
        type=int,
        default=8,
        help="Matching condition value used for max-projection visualization and pooled summary print.",
    )
    parser.add_argument(
        "--z_step", type=int, default=64, help="z chunk size for max-projection visualization"
    )
    parser.add_argument(
        "--z_damp",
        type=int,
        default=4,
        help="extra z padding (in slices) for TP/FP/FN points in each max-projection chunk",
    )
    parser.add_argument(
        "--compare_models", action="store_true", help="Set to True to set x axis as model name"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Mute stdout except the final pooled metrics and skip visualization",
    )
    return parser


def crop_range(i, input_size, volume_size):
    if (i + 1) * input_size > volume_size:
        start = volume_size - input_size
        end = volume_size
    else:
        start = i * input_size
        end = (i + 1) * input_size

    return start, end


def add_table_log(headers, lists, file_name):
    table = {}
    for header, list in zip(headers, lists):
        table[header] = list

    # Write the table to a text file
    with open(file_name, "a") as file:
        # Write the headers
        file.write("\t".join(headers) + "\n")
        # Write the rows
        for i in range(len(lists[0])):
            row = [str(table[key][i]) for key in headers]
            file.write("\t".join(row) + "\n")


def _safe_name(text):
    return "".join([c if c.isalnum() or c in ["-", "_", "."] else "_" for c in str(text)])


def _format_threshold_hm(threshold_hm):
    return f"{float(threshold_hm):g}"


def _threshold_hm_dir_name(threshold_hm):
    return _format_threshold_hm(threshold_hm).replace("0.", "0")


def _resolve_threshold_hms(threshold_hm, num_models):
    if isinstance(threshold_hm, (float, int)):
        threshold_hms = [float(threshold_hm)]
    else:
        threshold_hms = [float(t) for t in threshold_hm]

    if not threshold_hms:
        raise ValueError("--threshold_hm must include at least one value.")
    if len(threshold_hms) == 1:
        return threshold_hms * num_models, False
    if len(threshold_hms) != num_models:
        raise ValueError(
            f"--threshold_hm length ({len(threshold_hms)}) must be 1 or match "
            f"--models length ({num_models})."
        )
    return threshold_hms, True


def _build_volume_id(annotation, volume_idx):
    if not isinstance(annotation, dict):
        return f"vol_{volume_idx:03d}"
    brain_id = _safe_name(annotation.get("brain_id", "unknown_brain"))
    v_range = annotation.get("range", [])
    if isinstance(v_range, list) and len(v_range) == 2:
        (z0, y0, x0), (z1, y1, x1) = v_range
        return f"vol_{volume_idx:03d}_{brain_id}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}"
    return f"vol_{volume_idx:03d}_{brain_id}"


def _as_points(points):
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] > 3:
        arr = arr[:, :3]
    return arr


def _normalize_for_viz(image_2d, lower_percentile=1, upper_percentile=99):
    p_low, p_high = np.percentile(image_2d, [lower_percentile, upper_percentile])
    if p_high <= p_low:
        p_low, p_high = float(np.min(image_2d)), float(np.max(image_2d))
    if p_high <= p_low:
        return np.zeros_like(image_2d, dtype=np.float32)
    out = (image_2d - p_low) / (p_high - p_low + 1e-8)
    return np.clip(out, 0, 1).astype(np.float32)


def save_match_max_projection(
    voxel, tp_points, fn_points, fp_points, out_dir, z_step=64, z_damp=8, title_prefix=""
):
    if z_step <= 0:
        raise ValueError(f"z_step must be positive, got {z_step}")
    if z_damp < 0:
        raise ValueError(f"z_damp must be non-negative, got {z_damp}")

    if torch.is_tensor(voxel):
        volume = voxel.detach().cpu().numpy()
    else:
        volume = np.asarray(voxel)
    raw_shape = tuple(volume.shape)

    # Accept [B, C, Z, Y, X], [C, Z, Y, X], [Z, Y, X], or [Z, Y, X, C]
    if volume.ndim == 5:
        volume = volume[0]
    if volume.ndim == 3:
        volume = volume[np.newaxis, ...]
    elif volume.ndim == 4 and volume.shape[0] > 4 and volume.shape[-1] <= 4:
        volume = np.moveaxis(volume, -1, 0)
    if volume.ndim != 4:
        raise ValueError(f"Unexpected voxel shape for max projection: {volume.shape}")

    tp_points = _as_points(tp_points)
    fn_points = _as_points(fn_points)
    fp_points = _as_points(fp_points)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[save_match_max_projection] raw voxel shape={raw_shape}, "
        f"projection volume shape={tuple(volume.shape)} [C, Z, Y, X], "
        f"tp/fn/fp shapes={tp_points.shape}/{fn_points.shape}/{fp_points.shape}"
    )

    depth = volume.shape[1]
    for z0 in range(0, depth, z_step):
        z1 = min(z0 + z_step, depth)
        marker_channel = 1 if volume.shape[0] > 1 else 0
        gray_proj = _normalize_for_viz(np.max(volume[marker_channel, z0:z1], axis=0), 1, 99)
        print(
            f"[save_match_max_projection] z[{z0}:{z1}] "
            f"gray_proj={gray_proj.shape}, marker_channel={marker_channel}"
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(gray_proj, cmap="gray", vmin=0, vmax=1, interpolation="none")

        # Keep the image projection strict to [z0, z1), but optionally widen point overlays.
        point_z0 = max(0, z0 - z_damp)
        point_z1 = min(depth, z1 + z_damp)
        tp_mask = (tp_points[:, 0] >= point_z0) & (tp_points[:, 0] < point_z1)
        fn_mask = (fn_points[:, 0] >= point_z0) & (fn_points[:, 0] < point_z1)
        fp_mask = (fp_points[:, 0] >= point_z0) & (fp_points[:, 0] < point_z1)

        tp_chunk = tp_points[tp_mask]
        fn_chunk = fn_points[fn_mask]
        fp_chunk = fp_points[fp_mask]

        if tp_chunk.size > 0:
            ax.scatter(
                tp_chunk[:, 2],
                tp_chunk[:, 1],
                s=36,
                marker="o",
                c="yellow",
                edgecolors="black",
                linewidths=0.4,
                label=f"TP ({tp_chunk.shape[0]})",
            )
        if fp_chunk.size > 0:
            ax.scatter(
                fp_chunk[:, 2],
                fp_chunk[:, 1],
                s=100,
                marker="*",
                c="red",
                linewidths=0.8,
                label=f"FP ({fp_chunk.shape[0]})",
            )
        if fn_chunk.size > 0:
            ax.scatter(
                fn_chunk[:, 2],
                fn_chunk[:, 1],
                s=100,
                marker="*",
                c="dodgerblue",
                linewidths=0.8,
                label=f"FN ({fn_chunk.shape[0]})",
            )

        ax.set_title(f"{title_prefix} z[{z0}:{z1}]")
        ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                framealpha=0.6,
            )
        fig.tight_layout(rect=[0, 0, 0.82, 1], pad=0.1)
        fig.savefig(out_dir / f"maxproj_z{z0:04d}_{z1:04d}.png", dpi=180)
        plt.close(fig)


def _print_pooled_metrics(args, x_axis_label, x_s, pooled_summary_at_plot_r, stream=None):
    stream = stream or sys.stdout
    print(f"Pooled metrics at R={args.plot_R}:", file=stream, flush=True)
    threshold_hms = [
        item.get("threshold_hm")
        for item in pooled_summary_at_plot_r
        if item.get("threshold_hm") is not None
    ]
    has_per_model_thresholds = any(
        item.get("per_model_threshold_hm", False) for item in pooled_summary_at_plot_r
    )
    if threshold_hms and not has_per_model_thresholds:
        print(f"  threshold_hm={_format_threshold_hm(threshold_hms[0])}", file=stream, flush=True)
    elif threshold_hms:
        print("  threshold_hm=per-model", file=stream, flush=True)
    for item in pooled_summary_at_plot_r:
        metrics = item["metrics"]
        plot_label = item["plot_label"]
        if has_per_model_thresholds:
            plot_label = f'{plot_label} (threshold_hm={_format_threshold_hm(item["threshold_hm"])})'
        if metrics is None:
            print(
                f"  {plot_label}: not available (evaluated {x_axis_label} values: {x_s})",
                file=stream,
                flush=True,
            )
            continue
        print(
            f"  {plot_label}: "
            f'recall={metrics["recall_pooled"]:.6f}, '
            f'precision={metrics["precision_pooled"]:.6f}, '
            f'accuracy={metrics["accuracy_pooled"]:.6f}',
            file=stream,
            flush=True,
        )


def _main(args):

    assert len(args.R) == 1 or len(args.dist) == 1, "R and dist cant be ranges together"

    if not args.model_archs or not args.models or not args.plot_labels:
        raise ValueError("--model_archs, --models, and --plot_labels are required.")
    if len(args.model_archs) != len(args.models):
        raise ValueError(
            f"--model_archs length ({len(args.model_archs)}) must match "
            f"--models length ({len(args.models)})."
        )
    if len(args.plot_labels) != len(args.models):
        raise ValueError(
            f"--plot_labels length ({len(args.plot_labels)}) must match "
            f"--models length ({len(args.models)})."
        )
    threshold_hms, has_per_model_thresholds = _resolve_threshold_hms(
        args.threshold_hm, len(args.models)
    )

    if len(args.R) > 1:
        x_axis_label = "R"
        x_s = list(range(args.R[0], args.R[1] + 1, 2))
    else:
        x_axis_label = "dist"
        x_s = list(range(args.dist[0], args.dist[1] + 1))
    if args.plot_R in x_s:
        viz_match_x = args.plot_R
    else:
        viz_match_x = x_s[0]
        print(
            f"Warning: plot_R={args.plot_R} is not in evaluated {x_axis_label} list {x_s}. "
            f"Using {x_axis_label}={viz_match_x} for visualization."
        )

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d-%H%M", current_time)
    if has_per_model_thresholds:
        eval_run_dir = Path(args.eval_root) / args.testdata / timestamp
        threshold_hm_summary = [float(t) for t in threshold_hms]
        threshold_hm_title = "per-model"
    else:
        eval_run_dir = (
            Path(args.eval_root)
            / args.testdata
            / f"{_threshold_hm_dir_name(threshold_hms[0])}_{timestamp}"
        )
        threshold_hm_summary = float(threshold_hms[0])
        threshold_hm_title = _format_threshold_hm(threshold_hms[0])
    eval_viz_dir = eval_run_dir / "viz"
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_model = {
        "timestamp": timestamp,
        "testdata": args.testdata,
        "threshold_hm": threshold_hm_summary,
        "per_model_threshold_hm": bool(has_per_model_thresholds),
        "x_axis_label": x_axis_label,
        "matching_conditions": [int(x) for x in x_s],
        "plot_R": int(args.plot_R),
        "viz_match_condition": {x_axis_label: int(viz_match_x)},
        "z_step": int(args.z_step),
        "z_damp": int(args.z_damp),
        "models": {},
    }

    dataset = CellDetectionDatset(
        data_path=args.data_path,
        json_path=f"data/{args.testdata}.json",
        transforms=transforms,
        box_width=16,
        params=params,
        batched=False,
        R=args.downsample_ratio,
        labeled=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=None, batch_size=1, num_workers=1, pin_memory=True
    )

    # init log and savedir

    metrics_dir = Path(args.eval_root) / "metrics_plot" / args.testdata
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plt_filename = str(metrics_dir / timestamp)
    log_filepath = str(metrics_dir / "evaluate_log.txt")

    # Check if the file already exists
    if os.path.exists(log_filepath):
        print(f"File {log_filepath} already exists.")
    else:
        # Open the file in write mode ('w') and create it
        with open(log_filepath, "w") as file:
            file.write(f"log timestamp: {timestamp}")
        print(f"New file created at {log_filepath}")

    models = []
    checkpoints = []
    if not args.checkpoints:
        for i in range(len(args.model_archs)):
            checkpoints.append("checkpoint-199.pth")
    else:
        checkpoints = args.checkpoints
    if len(checkpoints) != len(args.models):
        raise ValueError(
            f"--checkpoints length ({len(checkpoints)}) must match "
            f"--models length ({len(args.models)})."
        )

    if args.ckpt_folders:
        if len(args.ckpt_folders) != len(args.models):
            raise ValueError(
                f"--ckpt_folders length ({len(args.ckpt_folders)}) must match "
                f"--models length ({len(args.models)})."
            )
        ckpt_folders = args.ckpt_folders
    else:
        ckpt_folders = [args.ckpt_folder] * len(args.models)

    output_dir = Path(args.ckpt_root)
    for model_arch, model_name, ckpt_folder, ckpt in zip(
        args.model_archs, args.models, ckpt_folders, checkpoints
    ):
        if args.pred_heads == "hm_off":
            model = models_mae_cn_hm_offset.__dict__[model_arch]()
        checkpoint_path = output_dir / ckpt_folder / f"brainmae-{args.encoder}-{model_name}" / ckpt
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)

    recalls = []
    precisions = []
    accuracys = []
    pooled_summary_at_plot_r = []

    for model_name, model, plot_label, ckpt_folder, ckpt, threshold_hm in zip(
        args.models, models, args.plot_labels, ckpt_folders, checkpoints, threshold_hms
    ):
        print(f"evaluating {model_name}...")
        tps, fps, fns = defaultdict(int), defaultdict(int), defaultdict(int)  # key is R or dist
        recalls_per_volume = defaultdict(list)
        precisions_per_volume = defaultdict(list)
        accuracies_per_volume = defaultdict(list)
        model_metrics = {
            "model_name": model_name,
            "plot_label": plot_label,
            "ckpt_folder": ckpt_folder,
            "checkpoint": ckpt,
            "threshold_hm": float(threshold_hm),
            "plot_R": int(args.plot_R),
            "overall": {},
            "per_volume": {},
        }
        pooled_metrics_at_plot_r = None
        with open(log_filepath, "a") as file:
            file.write(
                f"\n{model_name} evaluation on {args.testdata}. "
                f"threshold_hm={_format_threshold_hm(threshold_hm)}."
            )
        for vol_idx, (voxel, *gts, centers) in enumerate(dataloader):
            annotation = dataset.data[vol_idx] if vol_idx < len(dataset.data) else {}
            volume_id = _build_volume_id(annotation, vol_idx)
            print(f"Visualization input voxel tensor shape: {tuple(voxel.shape)}")
            if (
                isinstance(annotation, dict)
                and "range" in annotation
                and len(annotation["range"]) == 2
            ):
                (z0, y0, x0), (z1, y1, x1) = annotation["range"]
                print(
                    f"Annotation range: z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}] "
                    f"-> crop size {(z1 - z0, y1 - y0, x1 - x0)}"
                )
            centers = centers.squeeze(0).cpu().numpy()
            print(f"Number of gt centers in this vol: {centers.shape[0]}")
            if args.pred_heads == "hm_off":
                [z, y, x] = voxel.shape[-3:]
                whole_hm = np.zeros(
                    (
                        1,
                        z // args.downsample_ratio,
                        y // args.downsample_ratio,
                        x // args.downsample_ratio,
                    )
                )
                whole_off = np.zeros(
                    (
                        3,
                        z // args.downsample_ratio,
                        y // args.downsample_ratio,
                        x // args.downsample_ratio,
                    )
                )
                print(f"volume size: {z}x{y}x{x}. heatmap size: {whole_hm.shape}")

                with torch.no_grad():
                    for iz in range(z // args.input_size + 1):
                        for iy in range(y // args.input_size + 1):
                            for ix in range(x // args.input_size + 1):
                                start_z, end_z = crop_range(iz, args.input_size, z)
                                start_y, end_y = crop_range(iy, args.input_size, y)
                                start_x, end_x = crop_range(ix, args.input_size, x)
                                input_voxel = voxel[
                                    :, :, start_z:end_z, start_y:end_y, start_x:end_x
                                ]
                                start_time = time.time()
                                pred_hm, pred_off = model(input_voxel)
                                print(f"prediction done in {time.time() - start_time}")
                                pred_hm = pred_hm.detach().squeeze(0).cpu().numpy()
                                pred_off = pred_off.detach().squeeze(0).cpu().numpy()
                                if len(pred_hm.shape) == 3:
                                    print("Reshaping pred_hm")
                                    pred_hm = pred_hm[np.newaxis]
                                whole_hm[
                                    :,
                                    start_z
                                    // args.downsample_ratio : end_z
                                    // args.downsample_ratio,
                                    start_y
                                    // args.downsample_ratio : end_y
                                    // args.downsample_ratio,
                                    start_x
                                    // args.downsample_ratio : end_x
                                    // args.downsample_ratio,
                                ] = pred_hm
                                whole_off[
                                    :,
                                    start_z
                                    // args.downsample_ratio : end_z
                                    // args.downsample_ratio,
                                    start_y
                                    // args.downsample_ratio : end_y
                                    // args.downsample_ratio,
                                    start_x
                                    // args.downsample_ratio : end_x
                                    // args.downsample_ratio,
                                ] = pred_off

                # TODO: Change the model code to fix this shape matching issue
                # Right now it doesn't affect training because the channel dimension is 1 anyway
            else:
                raise ValueError("Invalid prediction heads.")

            volume_viz_dir = None
            if not getattr(args, "silent", False):
                viz2 = Visualizer(whole_hm, ["pred"])
                viz_hm = np.concatenate([np.array(v) for v in viz2.ind_rgbs.values()], axis=-2)
                model_viz_folder = _safe_name(plot_label)
                if has_per_model_thresholds:
                    model_viz_folder = (
                        f"{model_viz_folder}_hm{_threshold_hm_dir_name(threshold_hm)}"
                    )
                volume_viz_dir = eval_viz_dir / model_viz_folder / volume_id
                volume_viz_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(
                    str(volume_viz_dir / "pred_hm.gif"), [xy for xy in viz_hm], duration=0.3
                )

            volume_counts = {}
            viz_tp, viz_fn, viz_fp = [], [], []

            if x_axis_label == "R":
                get_centers = _cn_decode_one_cls(
                    whole_hm,
                    reg=whole_off,
                    min_distance=args.dist[0],
                    threshold_heat=threshold_hm,
                    exclude_border=False,
                )[:, :3]
                pred_centers = get_centers * args.downsample_ratio
                # TODO: save pred_centers as json
                if volume_viz_dir is not None:
                    output_json = volume_viz_dir / "pred_centers.json"
                    with open(output_json, "w") as f:
                        json.dump(pred_centers.tolist(), f, indent=2)

                print(f"Detected {len(pred_centers)}.")
                for r in x_s:
                    tp, fn, fp = match_cell(pred_centers, centers, R=r)
                    tps[r] += len(tp)
                    fns[r] += len(fn)
                    fps[r] += len(fp)
                    volume_counts[str(r)] = {
                        "tp": int(len(tp)),
                        "fn": int(len(fn)),
                        "fp": int(len(fp)),
                        "num_pred": int(len(pred_centers)),
                        "num_gt": int(centers.shape[0]),
                    }
                    if r == viz_match_x:
                        viz_tp, viz_fn, viz_fp = tp, fn, fp

            if x_axis_label == "dist":
                for d in x_s:
                    get_centers = _cn_decode_one_cls(
                        whole_hm,
                        reg=whole_off,
                        min_distance=d,
                        threshold_heat=threshold_hm,
                        exclude_border=False,
                    )[:, :3]
                    pred_centers = (get_centers + 1) * args.downsample_ratio

                    print(f"Detected {len(pred_centers)}.")
                    tp, fn, fp = match_cell(pred_centers, centers, R=args.R[0])
                    tps[d] += len(tp)
                    fns[d] += len(fn)
                    fps[d] += len(fp)
                    volume_counts[str(d)] = {
                        "tp": int(len(tp)),
                        "fn": int(len(fn)),
                        "fp": int(len(fp)),
                        "num_pred": int(len(pred_centers)),
                        "num_gt": int(centers.shape[0]),
                    }
                    if d == viz_match_x:
                        viz_tp, viz_fn, viz_fp = tp, fn, fp

            if volume_viz_dir is not None:
                proj_dir = volume_viz_dir / f"maxproj_{x_axis_label}_{viz_match_x}"
                save_match_max_projection(
                    voxel=voxel,
                    tp_points=viz_tp,
                    fn_points=viz_fn,
                    fp_points=viz_fp,
                    out_dir=proj_dir,
                    z_step=args.z_step,
                    z_damp=args.z_damp,
                    title_prefix=f"{plot_label} ({x_axis_label}={viz_match_x})",
                )

            model_metrics["per_volume"][volume_id] = {
                "volume_index": int(vol_idx),
                "brain_id": annotation.get("brain_id", ""),
                "range": annotation.get("range", []),
                "num_gt": int(centers.shape[0]),
                "matching_conditions": volume_counts,
                "viz_match_condition": {x_axis_label: int(viz_match_x)},
                "viz_dir": str(volume_viz_dir) if volume_viz_dir is not None else "",
            }
            for x in x_s:
                c = volume_counts.get(str(x), {"tp": 0, "fn": 0, "fp": 0})
                tp_i, fn_i, fp_i = c["tp"], c["fn"], c["fp"]
                rec_i = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
                prec_i = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
                acc_i = float(tp_i / (tp_i + fn_i + fp_i)) if (tp_i + fn_i + fp_i) > 0 else 0.0
                recalls_per_volume[x].append(rec_i)
                precisions_per_volume[x].append(prec_i)
                accuracies_per_volume[x].append(acc_i)

        recall, precision, accuracy = [], [], []

        for x in x_s:
            print(f"For {x_axis_label}={x}, TP: {tps[x]}, FN: {fns[x]}, FP: {fps[x]}")
            prec_pooled, rec_pooled, acc_pooled = evaluate_metrics(tps[x], fns[x], fps[x])
            rec_avg = float(np.mean(recalls_per_volume[x])) if recalls_per_volume[x] else 0.0
            prec_avg = float(np.mean(precisions_per_volume[x])) if precisions_per_volume[x] else 0.0
            acc_avg = float(np.mean(accuracies_per_volume[x])) if accuracies_per_volume[x] else 0.0
            recall.append(rec_avg)
            precision.append(prec_avg)
            accuracy.append(acc_avg)
            print(
                f"For {x_axis_label}={x}, "
                f"avg recall={rec_avg:.6f} (pooled={rec_pooled:.6f}), "
                f"avg precision={prec_avg:.6f} (pooled={prec_pooled:.6f}), "
                f"avg accuracy={acc_avg:.6f} (pooled={acc_pooled:.6f})"
            )
            model_metrics["overall"][str(x)] = {
                "tp": int(tps[x]),
                "fn": int(fns[x]),
                "fp": int(fps[x]),
                "precision": float(prec_avg),
                "recall": float(rec_avg),
                "accuracy": float(acc_avg),
                "precision_pooled": float(prec_pooled),
                "recall_pooled": float(rec_pooled),
                "accuracy_pooled": float(acc_pooled),
            }
            if x == args.plot_R:
                pooled_metrics_at_plot_r = {
                    "precision_pooled": float(prec_pooled),
                    "recall_pooled": float(rec_pooled),
                    "accuracy_pooled": float(acc_pooled),
                }

        recalls.append(recall)
        precisions.append(precision)
        accuracys.append(accuracy)
        model_metrics["pooled_at_plot_R"] = pooled_metrics_at_plot_r
        metrics_by_model["models"][plot_label] = model_metrics
        pooled_summary_at_plot_r.append(
            {
                "plot_label": plot_label,
                "model_name": model_name,
                "threshold_hm": float(threshold_hm),
                "per_model_threshold_hm": bool(has_per_model_thresholds),
                "metrics": pooled_metrics_at_plot_r,
            }
        )
        add_table_log(
            [x_axis_label, "recall", "precision", "accuracy"],
            [x_s, recall, precision, accuracy],
            log_filepath,
        )
        plt_filename += "_" + plot_label
    plt_filename += f"_{x_axis_label}.png"

    metrics_json_path = eval_run_dir / f"{plot_label}_tp_fn_fp_by_condition.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_by_model, f, indent=2)
    print(f"Saved tp/fn/fp json: {metrics_json_path}")

    if getattr(args, "silent", False):
        return x_axis_label, x_s, pooled_summary_at_plot_r

    if args.compare_models:
        # subplot
        fig, axs = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"test on {args.testdata}, \nhm_threshold: {threshold_hm_title}, \nx: models")

        x_labels = [name.split("_")[2] for name in args.plot_labels]
        x = np.arange(len(x_labels))

        for i, (title, y_s) in enumerate(
            zip(["recall", "precision", "accuracy"], [recalls, precisions, accuracys])
        ):
            y_values = []
            for j, (model_name, y) in enumerate(zip(args.plot_labels, y_s)):
                y = np.array(y)
                y_mean = np.mean(y)
                y_values.append(y_mean)
                print(f"avg {title} of {model_name}: {y_mean} ")

            axs.plot(
                x,
                np.array(y_values),
                label=title,
                color=list(mcolors.TABLEAU_COLORS.values())[i + 1],
            )
            axs.set_xlabel(x_axis_label)
            axs.set_xticks(x, labels=x_labels, fontsize=20)
            axs.legend()

    else:
        # subplot
        fig, axs = plt.subplots(3, 1, figsize=(6, 12))
        fig.suptitle(f"test on {args.testdata}, \nhm_threshold: {threshold_hm_title}", fontsize=18)

        for i, (title, y_s) in enumerate(
            zip(["recall", "precision", "accuracy"], [recalls, precisions, accuracys])
        ):
            x_labels = list(x_s)
            x = np.arange(len(x_labels))

            axs[i].set_title(title, fontsize=20)
            for j, (model_name, y) in enumerate(zip(args.plot_labels, y_s)):
                y = np.array(y)
                xnew = np.linspace(x.min(), x.max(), 100)
                spl = make_interp_spline(x, y, k=3)
                y_smooth = spl(xnew)
                if x_axis_label == "R":
                    axs[i].set_xlim(x[-1], x[0])
                axs[i].set_ylim(0.5, 1.025)
                axs[i].plot(
                    xnew,
                    y_smooth,
                    label=model_name,
                    color=list(mcolors.TABLEAU_COLORS.values())[j + 1],
                )
                axs[i].set_xlabel(x_axis_label, fontsize=16)
                axs[i].set_xticks(x, labels=x_labels, fontsize=12)
                for label in axs[i].get_yticklabels():
                    label.set_fontsize(12)

                axs[i].legend()
                print(f"avg {title} of {model_name}: {np.mean(y)} ")
    fig.tight_layout()
    fig.savefig(plt_filename)
    return x_axis_label, x_s, pooled_summary_at_plot_r


def main(args):
    if getattr(args, "silent", False):
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            x_axis_label, x_s, pooled_summary_at_plot_r = _main(args)
        _print_pooled_metrics(args, x_axis_label, x_s, pooled_summary_at_plot_r)
        return

    x_axis_label, x_s, pooled_summary_at_plot_r = _main(args)
    _print_pooled_metrics(args, x_axis_label, x_s, pooled_summary_at_plot_r)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
