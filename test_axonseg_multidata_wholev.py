import argparse
import torch
import numpy as np
import time
from util.axon_seg_datasets_hm_multidata import load_annotations_from_json, convert_annotations
from util.bmae_utils import (
    pretif_open,
    connect_neurite_points,
    skeletonize_pred,
    skeleton_to_simplified_paths,
)
from util.transforms import Transforms
from util.visualizer import Visualizer
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

import models_mae_axon_hm_multidata as models_mae_axon_hm
from models_mae_axon_hm_multidata import pred_to_boxes, assign_instance_ids


import json
import tifffile

import os
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.measure import label as cc_label
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score


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
        "--testdata", type=str, default="sparse_128_0", help="filename of test data in data folder"
    )
    parser.add_argument(
        "--fMOST", action="store_true", help="Set to True if the data is fMOST data"
    )
    parser.add_argument("--check_dataset", action="store_true", help="Set to True no test run")
    parser.add_argument("--labeled", action="store_true", help="Set to True to use labeled data")
    parser.add_argument(
        "--twochannel",
        action="store_true",
        help="Set to True to use 2-channel input for models trained with 2-channel input",
    )
    parser.add_argument(
        "--encoder", type=str, default="vitb", help="config of pretrained encoder, default: vitb"
    )
    parser.add_argument(
        "--output_patch_size",
        type=int,
        default=2,
        help="patch size of the MAE model output for hm pred, default: 2",
    )
    parser.add_argument(
        "--p_min", type=float, default=1.0, help="Lower percentile for normalize_by_percentile"
    )
    parser.add_argument(
        "--p_max", type=float, default=99.0, help="Upper percentile for normalize_by_percentile"
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
        "--input_size", nargs="+", type=int, default=[32, 256, 256], help="input size of the model"
    )
    parser.add_argument(
        "--overlap", nargs="+", type=int, default=[4, 16, 16], help="overlap between patches"
    )
    parser.add_argument("--viz_z_step", type=int, default=64, help="z-step for visualization")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        default=[],
        help="checkpoint of each model, same length as model_archs and model_name, if empty, then checkpoint-199.pth by default",
    )
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="/media/hddshare/general/mae_brain/brain-mae/output_dir/axon_hm",
        help="root folder containing brainmae-{encoder}-{model_name}/checkpoint files",
    )
    parser.add_argument(
        "--eval_root",
        default="/media/hddshare/general/mae_brain/brain-mae/eval/axonTrace",
        help="root for evaluation metrics and visualizations",
    )
    parser.add_argument(
        "--min_seg_length",
        type=int,
        default=5,
        help="Minimum length of skeleton segments to keep for evaluation and visualization",
    )
    parser.add_argument(
        "--thick_viz",
        action="store_true",
        help="Set to True to dilate the axon segments in viz_esa_distances visualization",
    )
    # Evaluation parameters
    parser.add_argument(
        "--hm_threshold",
        nargs="+",
        type=str,
        default=["0.1"],
        help="Heatmap threshold(s) for inference. Supports `--hm_threshold 0.1 0.2` or `--hm_threshold 0.1,0.2`.",
    )
    parser.add_argument("--selected_axon_id", nargs="+", type=str, default=None)
    parser.add_argument(
        "--instance_id_R", type=float, default=0.8, help="Radius for instance ID assignment"
    )
    parser.add_argument(
        "--merge_instance",
        action="store_true",
        help="Set to True to merge pred instances based on morphology overlap in one pred volume",
    )
    parser.add_argument(
        "--merge_dist",
        type=float,
        default=2.0,
        help="Distance threshold for merging instances in overlapping areas of tiling or in volume",
    )
    parser.add_argument(
        "--exclude_nuclei",
        type=float,
        default=0.0,
        help="Distance threshold to exclude nuclei regions during evaluation",
    )
    parser.add_argument(
        "--downsample_gt",
        action="store_true",
        help="Evaluate in prediction space by downsampling GT coordinates instead of upsampling predictions",
    )
    parser.add_argument(
        "--overlap_vote_k", type=int, default=2, help="K neighbors required to keep overlap points"
    )
    parser.add_argument(
        "--overlap_vote_ratio",
        type=float,
        default=0.4,
        help="Ratio threshold for instance merge after overlap vote",
    )
    parser.add_argument(
        "--overlap_vote_two_sides",
        action="store_true",
        help="Require support from >=2 directions if point lies in multiple overlaps",
    )
    parser.add_argument(
        "--match_cost_thr",
        type=float,
        default=None,
        help="Cost threshold for instance matching (default: eval_radius)",
    )
    parser.add_argument(
        "--match_tau_p",
        type=float,
        default=0.7,
        help="Pred coverage threshold for instance matching",
    )
    parser.add_argument(
        "--match_tau_g", type=float, default=0.7, help="GT coverage threshold for instance matching"
    )
    parser.add_argument(
        "--break_branch",
        action="store_true",
        help="Split each predicted instance into per-branch instances before evaluation",
    )
    parser.add_argument(
        "--min_overlap_length",
        type=int,
        default=5,
        help="Minimum overlap length for instance merging",
    )
    parser.add_argument(
        "--hm_whole_volume_skeleton",
        action="store_true",
        help="For heatmap models, build a whole-volume consensus heatmap, skeletonize it in output-grid space, then upsample the centerline back to voxel space.",
    )
    parser.add_argument(
        "--eval_radius",
        nargs="+",
        type=str,
        default=["4.0"],
        help="Distance threshold(s) (in voxels) for evaluation. Supports `--eval_radius 5 10` or `--eval_radius 5,10`.",
    )
    parser.add_argument(
        "--path_connectivity_distances",
        nargs="+",
        type=str,
        default=["5", "10", "25", "50", "100", "150"],
        help="GT path lengths as voxel counts for R-tolerant path connectivity. Supports `--path_connectivity_distances 25 50` or `--path_connectivity_distances 25,50`.",
    )
    return parser


def _as_pts(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Points must have shape (N,3). Got {a.shape}.")
    return a


def _prf(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def _radius_key(radius):
    return f"{float(radius):g}"


def _parse_eval_radii(values):
    if values is None:
        return [4.0]
    raw_values = values if isinstance(values, (list, tuple)) else [values]
    out = []
    seen = set()
    for raw in raw_values:
        if raw is None:
            continue
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            radius = float(token)
            if radius < 0:
                raise ValueError(f"eval_radius must be non-negative, got {radius}")
            key = _radius_key(radius)
            if key in seen:
                continue
            seen.add(key)
            out.append(radius)
    if not out:
        raise ValueError("eval_radius produced no valid radius values.")
    return out


def _parse_positive_distances(values, name):
    raw_values = values if isinstance(values, (list, tuple)) else [values]
    out = []
    seen = set()
    for raw in raw_values:
        if raw is None:
            continue
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            dist = float(token)
            if dist <= 0:
                raise ValueError(f"{name} must be positive, got {dist}")
            key = _radius_key(dist)
            if key in seen:
                continue
            seen.add(key)
            out.append(dist)
    if not out:
        raise ValueError(f"{name} produced no valid values.")
    return out


def _parse_hm_thresholds(values):
    if values is None:
        return [0.1]
    raw_values = values if isinstance(values, (list, tuple)) else [values]
    out = []
    seen = set()
    for raw in raw_values:
        if raw is None:
            continue
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            threshold = float(token)
            if not np.isfinite(threshold):
                raise ValueError(f"hm_threshold must be finite, got {threshold}")
            key = _radius_key(threshold)
            if key in seen:
                continue
            seen.add(key)
            out.append(threshold)
    if not out:
        raise ValueError("hm_threshold produced no valid values.")
    return out


def _build_eval_dirs(args, model_name, hm_threshold, timestamp):
    margin_suffix = "" if args.merge_instance else "_rawskeleton"
    hm_key = _radius_key(hm_threshold)
    base_dir = f"{args.eval_root}/viz/{args.testdata}_pred_{model_name}"
    if args.selected_axon_id is not None:
        selected_ids_str = "_".join(args.selected_axon_id)
        base_dir = f"{base_dir}/selected_id_{selected_ids_str}"
    if args.downsample_gt:
        base_dir = f"{base_dir}/downsample_gt"
    summary_viz_dir = (
        f"{base_dir}/conf_{hm_key}/instanceR_{args.instance_id_R}{margin_suffix}/{timestamp}"
    )
    hm_eval_dir = f"{base_dir}/instanceR_{args.instance_id_R}{margin_suffix}/{timestamp}"
    return summary_viz_dir, hm_eval_dir


def crop_range(i, input_size, volume_size):
    start = i * input_size
    end = min(start + input_size, volume_size)

    return start, end


def _axis_indices_with_tail_pad(origin, length, volume_size):
    raw = origin + np.arange(length, dtype=np.int64)
    idx = raw.copy()
    valid = (raw >= 0) & (raw < volume_size)

    tail_pos = np.where(raw >= volume_size)[0]
    pad_len = int(tail_pos.size)
    if pad_len > 0 and origin > 0:
        src = np.arange(origin - pad_len, origin, dtype=np.int64)
        src_valid = (src >= 0) & (src < volume_size)
        idx[tail_pos[src_valid]] = src[src_valid]
        valid[tail_pos[src_valid]] = True

    if volume_size > 0:
        safe_idx = np.clip(idx, 0, volume_size - 1)
    else:
        safe_idx = np.zeros_like(idx)
        valid[:] = False
    return safe_idx, valid


def _extract_patch_with_tail_pad(volume, origin_zyx, input_size_zyx):
    oz, oy, ox = [int(v) for v in origin_zyx]
    sz, sy, sx = [int(v) for v in input_size_zyx]
    z_size, y_size, x_size = [int(v) for v in volume.shape[-3:]]

    z_idx, z_valid = _axis_indices_with_tail_pad(oz, sz, z_size)
    y_idx, y_valid = _axis_indices_with_tail_pad(oy, sy, y_size)
    x_idx, x_valid = _axis_indices_with_tail_pad(ox, sx, x_size)

    patch = volume[:, z_idx][:, :, y_idx][:, :, :, x_idx].astype(np.float32, copy=False)
    valid_mask = z_valid[:, None, None] & y_valid[None, :, None] & x_valid[None, None, :]
    if not np.all(valid_mask):
        patch = patch.copy()
        patch *= valid_mask[None].astype(patch.dtype)
    return patch


def _filter_boxes_by_local_bounds(
    boxes, labels, logits, instance_ids, neuron_ids, input_size_zyx, valid_start_zyx, valid_end_zyx,
):
    if boxes.shape[0] == 0:
        return boxes, labels, logits, instance_ids, neuron_ids

    input_size_t = boxes.new_tensor(input_size_zyx)
    valid_start_t = boxes.new_tensor(valid_start_zyx)
    valid_end_t = boxes.new_tensor(valid_end_zyx)
    centers_local = boxes[:, :3] * input_size_t
    keep = ((centers_local >= valid_start_t) & (centers_local < valid_end_t)).all(dim=1)
    if keep.all():
        return boxes, labels, logits, instance_ids, neuron_ids
    return boxes[keep], labels[keep], logits[keep], instance_ids[keep], neuron_ids[keep]


def viz(out_dir, voxel, target, pred=None, segs=None, step_size=32):
    while voxel.ndim < 5:
        voxel = voxel.unsqueeze(0)
    viz2 = Visualizer(voxel.squeeze(0), ["raw"])
    target_list = _gt_segments_from_target(target)
    depth, height, width = voxel.shape[-3:]
    scale = np.array([depth, height, width], dtype=float)
    # list to dict by neurite id
    target_dict = defaultdict(list)
    for neurite_id, p0, p1 in target_list:
        target_dict[neurite_id].append((p0 / scale, p1 / scale))
    viz2.draw_axon_segments(target_dict, "gt_segs")
    if pred is not None:
        viz2.draw_axon_bbox(pred, "pred")
    if segs is not None:
        viz2.draw_axon_segments(segs, "pred_segs")
    viz2.write_max_projection_png(out_dir, step_size=step_size)


def save_data(dir, voxel, target, pred=None):
    def _to_serializable(obj):
        if torch.is_tensor(obj):
            return obj.cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    out_dir = f'{dir}/{dir.split("/")[-1]}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    volume_dir = os.path.join(out_dir, "vol")
    if not os.path.exists(volume_dir):
        os.makedirs(volume_dir)
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.squeeze().cpu().numpy()
    elif isinstance(voxel, np.ndarray):
        voxel = voxel.squeeze()
    # save target and pred dict as json
    serializable_tgt = {k: _to_serializable(v) for k, v in target.items()}
    target_path = os.path.join(out_dir, "gt.json")
    with open(target_path, "w") as f:
        json.dump(serializable_tgt, f)

    if pred is not None:
        pred = {k: _to_serializable(v) for k, v in pred.items()}
        pred_path = os.path.join(out_dir, "pred.json")
        with open(pred_path, "w") as f:
            json.dump(pred, f)

    # save each xy slice of voxel as tiff file
    for i in range(voxel.shape[0]):
        slice_path = os.path.join(volume_dir, f"{i:03d}.tiff")
        tifffile.imwrite(slice_path, voxel[i, :, :])


def check_dataset(data_path, data_vols, selected_axon_id=None):
    # dataset check
    for i, s in enumerate(data_vols):
        if "pretif" in s["subvol_folder"]:
            voxel = pretif_open(data_path / s["brain_id"] / s["axon_key"] / s["subvol_folder"])
        elif ".tif" in s["subvol_folder"]:
            voxel = tifffile.imread(data_path / s["brain_id"] / s["subvol_folder"])
        target = s["target"]
        if len(target) == 1:
            target = target[0]
        print(f"\n\ndataloader: {i}. Max intensity in the vol: {torch.max(voxel)}")
        out_dir = (
            f"{args.eval_root}/data_checks/{args.testdata}_gt/data_{i}"
            if selected_axon_id is None
            else f"{args.eval_root}/data_checks/{args.testdata}_axon_of_{selected_axon_id}_gt/data_{i}"
        )
        viz(out_dir, voxel, target, step_size=32)


def _as_point_coords(points, dtype=float):
    if points is None:
        return None
    if torch.is_tensor(points):
        arr = points.detach().cpu().numpy()
    else:
        arr = np.asarray(points)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=dtype if dtype is not None else float)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected point array with shape (N, >=3), got {arr.shape}")
    arr = arr[:, :3]
    return arr


def _flatten_center_sets(center_sets):
    if center_sets is None:
        return np.zeros((0, 3), dtype=float)

    if torch.is_tensor(center_sets) or isinstance(center_sets, np.ndarray):
        center_sets = [center_sets]
    elif not isinstance(center_sets, (list, tuple)):
        center_sets = [center_sets]

    out = []
    for centers in center_sets:
        if centers is None:
            continue
        arr = _as_point_coords(centers, dtype=float)
        if arr is None or arr.size == 0:
            continue
        out.append(arr)

    if out:
        return np.concatenate(out, axis=0)
    return np.zeros((0, 3), dtype=float)


def _get_exclusion_centers(target):
    annotations = target.get("annotations", {})
    return _flatten_center_sets(annotations.get("cell_centers"))


def _points_outside_radius_mask(points, ref_points, radius):
    pts = _as_point_coords(points, dtype=float)
    if pts is None:
        return None
    if pts.size == 0:
        return np.ones((0,), dtype=bool)

    radius = float(radius)
    if radius <= 0:
        return np.ones((pts.shape[0],), dtype=bool)

    refs = _as_point_coords(ref_points, dtype=float)
    if refs is None or refs.size == 0:
        return np.ones((pts.shape[0],), dtype=bool)

    ref_tree = cKDTree(refs)
    return ref_tree.query(pts, k=1)[0] > radius


def _downsample_points_to_grid(points, coord_scale=1.0):
    scale = float(coord_scale)
    if scale <= 0:
        raise ValueError(f"coord_scale must be positive, got {coord_scale}")
    arr = _as_point_coords(points, dtype=float)
    if arr is None:
        return None
    if arr.size == 0:
        return np.zeros((0, 3), dtype=int)
    return np.floor(arr / scale).astype(int, copy=False)


def _dedupe_labeled_points(points, inst_ids, neuron_ids):
    pts = _as_point_coords(points, dtype=int)
    if pts is None or pts.size == 0:
        return (
            np.zeros((0, 3), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    inst_arr = _as_numpy(inst_ids, dtype=int)
    neur_arr = _as_numpy(neuron_ids, dtype=int)
    if inst_arr is None or inst_arr.size != pts.shape[0]:
        inst_arr = np.full((pts.shape[0],), -1, dtype=int)
    if neur_arr is None or neur_arr.size != pts.shape[0]:
        neur_arr = np.full((pts.shape[0],), -1, dtype=int)

    rows = np.concatenate([pts, inst_arr[:, None], neur_arr[:, None]], axis=1)
    _, keep_idx = np.unique(rows, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    return pts[keep_idx], inst_arr[keep_idx], neur_arr[keep_idx]


def extract_gt_points(target: dict, coord_scale: float = 1.0, downsample_to_grid: bool = False):
    """Flatten neurite annotations into arrays of GT points and ids."""
    annotations = target.get("annotations", {})
    neurite_dicts = annotations.get("gt_neurites", [])
    neuron_maps = annotations.get("gt_neurons", [])

    pts_list, inst_ids, neuron_ids = [], [], []
    for idx, neurite_dict in enumerate(neurite_dicts):
        if not isinstance(neurite_dict, dict):
            continue
        neuron_map = neuron_maps[idx] if idx < len(neuron_maps) else {}
        for neurite_id, pts in neurite_dict.items():
            if pts is None:
                continue
            arr = connect_neurite_points(pts)
            if arr.size == 0:
                continue
            pts_list.append(arr)
            inst_ids.append(np.full(arr.shape[0], int(neurite_id), dtype=int))
            neuron_id = -1
            if isinstance(neuron_map, dict) and neurite_id in neuron_map:
                neuron_id = int(neuron_map[neurite_id])
            neuron_ids.append(np.full(arr.shape[0], neuron_id, dtype=int))

    if not pts_list:
        points = _gt_skeleton_points_from_target(
            target, coord_scale=coord_scale, downsample_to_grid=downsample_to_grid,
        )
        if points.size > 0:
            points, inst_arr = _connected_component_labels_from_points(points)
            neuron_arr = np.full((points.shape[0],), -1, dtype=int)
            return points, inst_arr, neuron_arr
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    points = np.concatenate(pts_list, axis=0).astype(float)
    inst_arr = np.concatenate(inst_ids, axis=0)
    neuron_arr = np.concatenate(neuron_ids, axis=0)
    if downsample_to_grid:
        points = _downsample_points_to_grid(points, coord_scale=coord_scale)
        points, inst_arr, neuron_arr = _dedupe_labeled_points(points, inst_arr, neuron_arr)
    return points, inst_arr, neuron_arr


def _dedupe_consecutive_points(points):
    pts = _as_point_coords(points, dtype=float)
    if pts is None or pts.size == 0:
        return np.zeros((0, 3), dtype=float)
    if pts.shape[0] <= 1:
        return pts.astype(float, copy=False)
    keep = np.ones((pts.shape[0],), dtype=bool)
    keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
    return pts[keep].astype(float, copy=False)


def _points_from_gt_value(value, connect_ordered=False):
    if value is None:
        return np.zeros((0, 3), dtype=float)
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    if arr.ndim == 4 and arr.shape[-1] != 3:
        if arr.shape[0] == 1:
            return np.argwhere(arr[0] > 0).astype(float, copy=False)
        if arr.shape[-1] == 1:
            return np.argwhere(arr[..., 0] > 0).astype(float, copy=False)
    if arr.ndim > 3 and arr.shape[-1] != 3:
        mask = np.squeeze(arr)
        if mask.ndim == 3:
            return np.argwhere(mask > 0).astype(float, copy=False)
    if arr.ndim == 3:
        return np.argwhere(arr > 0).astype(float, copy=False)

    pts = _as_point_coords(arr, dtype=float)
    if pts is None or pts.size == 0:
        return np.zeros((0, 3), dtype=float)
    if connect_ordered and pts.shape[0] >= 2:
        pts = connect_neurite_points(pts)
    return _dedupe_consecutive_points(pts)


def _gt_skeleton_points_from_target(
    target: dict, coord_scale: float = 1.0, downsample_to_grid: bool = False
):
    """Build a GT skeleton point cloud without relying on annotation instance IDs."""
    annotations = target.get("annotations", {})
    point_sets = []

    gt_neurites = annotations.get("gt_neurites", [])
    if isinstance(gt_neurites, dict):
        gt_neurites = [gt_neurites]
    if isinstance(gt_neurites, (list, tuple)):
        for neurite_source in gt_neurites:
            if isinstance(neurite_source, dict):
                for pts in neurite_source.values():
                    arr = _points_from_gt_value(pts, connect_ordered=True)
                    if arr.size:
                        point_sets.append(arr)
            else:
                arr = _points_from_gt_value(neurite_source, connect_ordered=True)
                if arr.size:
                    point_sets.append(arr)

    for container in (annotations, target):
        for key in (
            "gt_skeleton",
            "gt_skeleton_points",
            "gt_points",
            "skeleton",
            "skeleton_points",
        ):
            if key not in container:
                continue
            arr = _points_from_gt_value(container.get(key), connect_ordered=False)
            if arr.size:
                point_sets.append(arr)

    if not point_sets:
        return np.zeros((0, 3), dtype=float)

    points = np.concatenate(point_sets, axis=0).astype(float, copy=False)
    if downsample_to_grid:
        points = _downsample_points_to_grid(points, coord_scale=coord_scale)
    points = np.rint(points).astype(int, copy=False)
    points = np.unique(points, axis=0)
    return points.astype(float, copy=False)


def _prediction_points_for_connectivity(
    pred: dict, coord_scale: float = 1.0, downsample_to_grid: bool = False
):
    pred_centers = _as_point_coords(pred.get("centers"), dtype=float)
    if pred_centers is None or pred_centers.size == 0:
        return np.zeros((0, 3), dtype=float)
    if downsample_to_grid:
        pred_centers = _downsample_points_to_grid(pred_centers, coord_scale=coord_scale)
    pred_points = np.rint(pred_centers).astype(int, copy=False)
    pred_points = np.unique(pred_points, axis=0)
    return pred_points.astype(float, copy=False)


def _build_skeleton_adjacency(points):
    pts = _as_point_coords(points, dtype=float)
    if pts is None or pts.size == 0:
        return np.zeros((0, 3), dtype=float), []

    pts_int = np.rint(pts).astype(int, copy=False)
    pts_int = np.unique(pts_int, axis=0)
    n_pts = int(pts_int.shape[0])
    if n_pts == 0:
        return np.zeros((0, 3), dtype=float), []

    adjacency = [set() for _ in range(n_pts)]
    if n_pts > 1:
        tree = cKDTree(pts_int.astype(float, copy=False))
        for i, j in tree.query_pairs(r=np.sqrt(3.0) + 1e-6):
            i = int(i)
            j = int(j)
            if np.max(np.abs(pts_int[i] - pts_int[j])) <= 1:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return pts_int.astype(float, copy=False), [sorted(neighbors) for neighbors in adjacency]


def _component_labels_from_adjacency(adjacency):
    n_pts = len(adjacency)
    if n_pts == 0:
        return np.zeros((0,), dtype=int)
    parent = np.arange(n_pts, dtype=int)
    rank = np.zeros((n_pts,), dtype=np.uint8)

    def find_root(idx):
        idx = int(idx)
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a, b):
        ra = find_root(a)
        rb = find_root(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if int(j) > i:
                union(i, j)

    roots = np.asarray([find_root(i) for i in range(n_pts)], dtype=int)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(int, copy=False)


def _connected_component_labels_from_points(points):
    pts, adjacency = _build_skeleton_adjacency(points)
    return pts, _component_labels_from_adjacency(adjacency)


def _edge_key(i, j):
    i = int(i)
    j = int(j)
    return (i, j) if i < j else (j, i)


def _trace_skeleton_paths(points, adjacency, component_labels):
    points = _as_point_coords(points, dtype=float)
    if points is None or points.size == 0:
        return []

    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=int)
    visited_edges = set()
    paths = []

    def add_path(node_ids):
        if not node_ids:
            return
        path_points = points[np.asarray(node_ids, dtype=int)]
        path_points = _dedupe_consecutive_points(path_points)
        if path_points.size == 0:
            return
        paths.append(
            {
                "path_index": int(len(paths)),
                "component_id": int(component_labels[int(node_ids[0])]),
                "points": path_points,
            }
        )

    for idx, degree in enumerate(degrees):
        if degree == 0:
            add_path([idx])

    junction_nodes = np.flatnonzero(degrees != 2)
    for start in junction_nodes.tolist():
        if degrees[start] == 0:
            continue
        for first_neighbor in adjacency[start]:
            edge = _edge_key(start, first_neighbor)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            node_ids = [int(start), int(first_neighbor)]
            prev = int(start)
            cur = int(first_neighbor)
            while degrees[cur] == 2:
                next_nodes = [int(n) for n in adjacency[cur] if int(n) != prev]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                edge = _edge_key(cur, nxt)
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                node_ids.append(nxt)
                prev, cur = cur, nxt
            add_path(node_ids)

    for start in range(points.shape[0]):
        for first_neighbor in adjacency[start]:
            edge = _edge_key(start, first_neighbor)
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            node_ids = [int(start), int(first_neighbor)]
            prev = int(start)
            cur = int(first_neighbor)
            while True:
                next_nodes = [int(n) for n in adjacency[cur] if int(n) != prev]
                nxt = None
                for candidate in next_nodes:
                    if _edge_key(cur, candidate) not in visited_edges:
                        nxt = candidate
                        break
                if nxt is None:
                    break
                visited_edges.add(_edge_key(cur, nxt))
                node_ids.append(nxt)
                if nxt == start:
                    break
                prev, cur = cur, nxt
            add_path(node_ids)

    return paths


def _gt_skeleton_paths_from_target(
    target: dict, coord_scale: float = 1.0, downsample_to_grid: bool = False
):
    gt_points = _gt_skeleton_points_from_target(
        target, coord_scale=coord_scale, downsample_to_grid=downsample_to_grid,
    )
    gt_points, adjacency = _build_skeleton_adjacency(gt_points)
    component_labels = _component_labels_from_adjacency(adjacency)
    paths = _trace_skeleton_paths(gt_points, adjacency, component_labels)
    num_components = int(np.unique(component_labels).size) if component_labels.size > 0 else 0
    return paths, gt_points, component_labels, num_components


def _map_points_to_pred_components(points, pred_tree, pred_component_ids, radius):
    pts = _as_point_coords(points, dtype=float)
    n_pts = 0 if pts is None else int(pts.shape[0])
    comp_ids = np.full((n_pts,), -1, dtype=int)
    dists = np.full((n_pts,), float("inf"), dtype=float)
    if n_pts == 0 or pred_tree is None or pred_component_ids.size == 0:
        return comp_ids, dists

    dists, nn_idx = pred_tree.query(pts, k=1)
    dists = np.asarray(dists, dtype=float)
    nn_idx = np.asarray(nn_idx, dtype=int)
    matched = dists <= float(radius)
    if matched.any():
        comp_ids[matched] = pred_component_ids[nn_idx[matched]]
    return comp_ids, dists


def _path_edge_lengths(path_points):
    pts = _as_point_coords(path_points, dtype=float)
    if pts is None or pts.shape[0] < 2:
        return np.zeros((0,), dtype=float)
    edge_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    edge_lengths = np.where(np.isfinite(edge_lengths), edge_lengths, 0.0)
    return edge_lengths.astype(float, copy=False)


def _summarize_skeleton_component_lengths(
    lengths,
    point_counts=None,
    skeleton_ids=None,
    skeleton_id_key="component_ids",
    length_source="connected_components",
):
    lengths = np.asarray(lengths, dtype=float)
    finite_lengths = lengths[np.isfinite(lengths)]
    skeleton_lengths = lengths.astype(int, copy=False).tolist()
    payload = {
        "unit": "voxels",
        "length_definition": "count_unique_voxels_in_densified_skeleton",
        "length_source": str(length_source),
        "num_skeletons": int(lengths.size),
        "num_components": int(lengths.size),
        "total_length": float(np.sum(finite_lengths)) if finite_lengths.size else 0.0,
        "mean_length": float(np.mean(finite_lengths)) if finite_lengths.size else float("nan"),
        "median_length": float(np.median(finite_lengths)) if finite_lengths.size else float("nan"),
        "min_length": float(np.min(finite_lengths)) if finite_lengths.size else float("nan"),
        "max_length": float(np.max(finite_lengths)) if finite_lengths.size else float("nan"),
        "skeleton_lengths": skeleton_lengths,
        "component_lengths": skeleton_lengths,
    }
    if point_counts is not None:
        payload["point_counts"] = np.asarray(point_counts, dtype=int).tolist()
    if skeleton_ids is not None:
        payload[skeleton_id_key] = [
            int(v) if isinstance(v, (np.integer, int)) else v for v in skeleton_ids
        ]
    return payload


def _skeleton_component_lengths_from_points(points):
    pts, adjacency = _build_skeleton_adjacency(points)
    component_labels = _component_labels_from_adjacency(adjacency)
    if pts.size == 0 or component_labels.size == 0:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            [],
        )

    _, component_index = np.unique(component_labels, return_inverse=True)
    num_components = int(component_index.max()) + 1
    point_counts = np.bincount(component_index, minlength=num_components).astype(int, copy=False)
    lengths = point_counts.copy()
    component_ids = list(range(num_components))
    return lengths, point_counts, component_ids


def _gt_instance_skeleton_lengths_from_target(
    target: dict, coord_scale: float = 1.0, downsample_to_grid: bool = False,
):
    annotations = target.get("annotations", {})
    neurite_dicts = annotations.get("gt_neurites", [])
    neuron_maps = annotations.get("gt_neurons", [])
    if isinstance(neurite_dicts, dict):
        neurite_dicts = [neurite_dicts]

    lengths = []
    point_counts = []
    instance_ids = []
    neuron_ids = []
    for idx, neurite_dict in enumerate(neurite_dicts):
        if not isinstance(neurite_dict, dict):
            continue
        neuron_map = neuron_maps[idx] if idx < len(neuron_maps) else {}
        for neurite_id, pts in neurite_dict.items():
            arr = _points_from_gt_value(pts, connect_ordered=True)
            if arr.size == 0:
                continue
            if downsample_to_grid:
                arr = _downsample_points_to_grid(arr, coord_scale=coord_scale)
            arr = np.rint(arr).astype(int, copy=False)
            arr = np.unique(arr, axis=0)
            voxel_count = int(arr.shape[0])
            lengths.append(voxel_count)
            point_counts.append(voxel_count)
            instance_ids.append(int(neurite_id))
            neuron_id = -1
            if isinstance(neuron_map, dict) and neurite_id in neuron_map:
                neuron_id = int(neuron_map[neurite_id])
            neuron_ids.append(neuron_id)

    return (
        np.asarray(lengths, dtype=int),
        np.asarray(point_counts, dtype=int),
        instance_ids,
        neuron_ids,
    )


def _skeleton_component_length_stats_from_target(
    target: dict, coord_scale: float = 1.0, downsample_to_grid: bool = False,
):
    lengths, point_counts, instance_ids, neuron_ids = _gt_instance_skeleton_lengths_from_target(
        target, coord_scale=coord_scale, downsample_to_grid=downsample_to_grid,
    )
    if lengths.size > 0:
        stats = _summarize_skeleton_component_lengths(
            lengths,
            point_counts,
            skeleton_ids=instance_ids,
            skeleton_id_key="instance_ids",
            length_source="densified_gt_instances",
        )
        stats["neuron_ids"] = neuron_ids
        return stats

    points = _gt_skeleton_points_from_target(
        target, coord_scale=coord_scale, downsample_to_grid=downsample_to_grid,
    )
    lengths, point_counts, component_ids = _skeleton_component_lengths_from_points(points)
    return _summarize_skeleton_component_lengths(
        lengths,
        point_counts,
        skeleton_ids=component_ids,
        skeleton_id_key="component_ids",
        length_source="connected_components",
    )


def _skeleton_26cc_length_stats_from_target(
    target: dict, coord_scale: float = 1.0, downsample_to_grid: bool = False,
):
    points = _gt_skeleton_points_from_target(
        target, coord_scale=coord_scale, downsample_to_grid=downsample_to_grid,
    )
    lengths, point_counts, component_ids = _skeleton_component_lengths_from_points(points)
    return _summarize_skeleton_component_lengths(
        lengths,
        point_counts,
        skeleton_ids=component_ids,
        skeleton_id_key="component_ids",
        length_source="connected_components_26",
    )


def _accumulate_erl_counts(erl_counts, path_points, mapped_component_ids):
    pts = _as_point_coords(path_points, dtype=float)
    if pts is None or pts.size == 0:
        return

    comp_ids = np.asarray(mapped_component_ids, dtype=int)
    n_pts = int(pts.shape[0])
    erl_counts["num_paths"] += 1
    erl_counts["num_gt_points"] += n_pts
    erl_counts["num_mapped_gt_points"] += int(np.sum(comp_ids >= 0))

    if n_pts == 1:
        total_length = 1.0
        path_sum_sq = 1.0 if comp_ids[0] >= 0 else 0.0
        erl_counts["total_gt_length"] += total_length
        erl_counts["sum_run_length_sq"] += path_sum_sq
        erl_counts["normalized_run_length_sum"] += path_sum_sq / total_length
        if path_sum_sq > 0:
            erl_counts["num_runs"] += 1
            erl_counts["num_mapped_edges"] += 1
        erl_counts["num_edges"] += 1
        return

    edge_lengths = _path_edge_lengths(pts)
    valid_edges = edge_lengths > 0
    total_length = float(np.sum(edge_lengths[valid_edges]))
    if total_length <= 0:
        return

    erl_counts["total_gt_length"] += total_length
    erl_counts["num_edges"] += int(np.sum(valid_edges))

    run_component = -1
    run_length = 0.0
    path_sum_sq = 0.0

    def flush_run():
        nonlocal run_component, run_length, path_sum_sq
        if run_component >= 0 and run_length > 0:
            path_sum_sq += run_length * run_length
            erl_counts["num_runs"] += 1
        run_component = -1
        run_length = 0.0

    for edge_length, c0, c1 in zip(edge_lengths, comp_ids[:-1], comp_ids[1:]):
        if edge_length <= 0:
            continue
        edge_component = int(c0) if c0 >= 0 and c0 == c1 else -1
        if edge_component >= 0:
            erl_counts["num_mapped_edges"] += 1
            if run_component == edge_component:
                run_length += float(edge_length)
            else:
                flush_run()
                run_component = edge_component
                run_length = float(edge_length)
        else:
            flush_run()
    flush_run()

    erl_counts["sum_run_length_sq"] += path_sum_sq
    erl_counts["normalized_run_length_sum"] += path_sum_sq / total_length


def _accumulate_path_connectivity_counts(
    path_counts, path_points, mapped_component_ids, path_length_items
):
    pts = _as_point_coords(path_points, dtype=float)
    if pts is None or pts.shape[0] < 2:
        return

    comp_ids = np.asarray(mapped_component_ids, dtype=int)
    n_pts = int(pts.shape[0])
    if comp_ids.size != n_pts:
        return

    for d_key, path_distance in path_length_items:
        path_voxel_count = int(np.ceil(float(path_distance)))
        if path_voxel_count <= 0 or n_pts < path_voxel_count:
            continue
        counts = path_counts[d_key]
        if path_voxel_count == 1:
            for start_idx in range(n_pts):
                counts["total"] += 1
                if comp_ids[start_idx] >= 0:
                    counts["connected"] += 1
            continue
        for start_idx in range(n_pts - path_voxel_count + 1):
            end_idx = start_idx + path_voxel_count - 1
            counts["total"] += 1
            if comp_ids[start_idx] >= 0 and comp_ids[end_idx] == comp_ids[start_idx]:
                counts["connected"] += 1


def _new_r_tolerant_metric_counts(path_distance_keys):
    return {
        "erl": {
            "sum_run_length_sq": 0.0,
            "normalized_run_length_sum": 0.0,
            "total_gt_length": 0.0,
            "num_runs": 0,
            "num_paths": 0,
            "num_gt_points": 0,
            "num_mapped_gt_points": 0,
            "num_edges": 0,
            "num_mapped_edges": 0,
            "num_gt_components": 0,
            "num_pred_components": 0,
        },
        "path_connectivity": {d_key: {"connected": 0, "total": 0} for d_key in path_distance_keys},
    }


def r_tolerant_erl_and_path_connectivity(
    pred: dict,
    target: dict,
    radius_items,
    path_length_items,
    coord_scale: float = 1.0,
    downsample_to_grid: bool = False,
):
    """Compute R-tolerant ERL and path connectivity from physical pred CCs."""
    radius_items = list(radius_items)
    path_length_items = list(path_length_items)
    path_distance_keys = [d_key for d_key, _ in path_length_items]
    results = {
        r_key: _new_r_tolerant_metric_counts(path_distance_keys) for r_key, _ in radius_items
    }

    gt_paths, _, _, num_gt_components = _gt_skeleton_paths_from_target(
        target, coord_scale=coord_scale, downsample_to_grid=downsample_to_grid,
    )
    pred_points = _prediction_points_for_connectivity(
        pred, coord_scale=coord_scale, downsample_to_grid=downsample_to_grid,
    )
    pred_points, pred_component_ids = _connected_component_labels_from_points(pred_points)
    pred_tree = cKDTree(pred_points) if pred_points.size > 0 else None
    num_pred_components = (
        int(np.unique(pred_component_ids).size) if pred_component_ids.size > 0 else 0
    )

    for r_key, radius in radius_items:
        counts = results[r_key]
        counts["erl"]["num_gt_components"] = num_gt_components
        counts["erl"]["num_pred_components"] = num_pred_components
        for path_info in gt_paths:
            path_points = path_info["points"]
            mapped_component_ids, _ = _map_points_to_pred_components(
                path_points, pred_tree, pred_component_ids, radius,
            )
            _accumulate_erl_counts(counts["erl"], path_points, mapped_component_ids)
            _accumulate_path_connectivity_counts(
                counts["path_connectivity"], path_points, mapped_component_ids, path_length_items,
            )

    return results


def _accumulate_r_tolerant_metric_counts(total_counts, volume_counts):
    for r_key, counts in volume_counts.items():
        if r_key not in total_counts:
            continue
        dst_erl = total_counts[r_key]["erl"]
        src_erl = counts.get("erl", {})
        for key in (
            "sum_run_length_sq",
            "normalized_run_length_sum",
            "total_gt_length",
            "num_runs",
            "num_paths",
            "num_gt_points",
            "num_mapped_gt_points",
            "num_edges",
            "num_mapped_edges",
            "num_gt_components",
            "num_pred_components",
        ):
            dst_erl[key] += src_erl.get(key, 0)

        for d_key, src_path_counts in counts.get("path_connectivity", {}).items():
            if d_key not in total_counts[r_key]["path_connectivity"]:
                continue
            dst_path_counts = total_counts[r_key]["path_connectivity"][d_key]
            dst_path_counts["connected"] += int(src_path_counts.get("connected", 0))
            dst_path_counts["total"] += int(src_path_counts.get("total", 0))


def _finalize_r_tolerant_metric_counts(total_counts, path_distances, length_scale: float = 1.0):
    length_scale = float(length_scale)
    length_scale_sq = length_scale * length_scale
    path_distance_keys = [_radius_key(d) for d in path_distances]
    erl_payload = {
        "normalized": {},
        "expected_run_length": {},
        "total_gt_length": {},
        "sum_run_length_sq": {},
        "num_runs": {},
        "num_paths": {},
        "num_gt_points": {},
        "num_mapped_gt_points": {},
        "num_edges": {},
        "num_mapped_edges": {},
        "num_gt_components": {},
        "num_pred_components": {},
    }
    path_payload = {
        "distances": [float(d) for d in path_distances],
        "distance_unit": "voxels",
        "distance_definition": "number_of_voxels_along_26_connected_gt_path",
        "score": {},
        "connected": {},
        "total": {},
    }

    for r_key, counts in total_counts.items():
        erl_counts = counts["erl"]
        total_gt_length = float(erl_counts["total_gt_length"])
        sum_run_length_sq = float(erl_counts["sum_run_length_sq"])
        normalized_sum = float(erl_counts["normalized_run_length_sum"])

        erl_payload["normalized"][r_key] = (
            float(normalized_sum / total_gt_length) if total_gt_length > 0 else float("nan")
        )
        erl_payload["expected_run_length"][r_key] = (
            float((sum_run_length_sq / total_gt_length) * length_scale)
            if total_gt_length > 0
            else float("nan")
        )
        erl_payload["total_gt_length"][r_key] = float(total_gt_length * length_scale)
        erl_payload["sum_run_length_sq"][r_key] = float(sum_run_length_sq * length_scale_sq)
        for key in (
            "num_runs",
            "num_paths",
            "num_gt_points",
            "num_mapped_gt_points",
            "num_edges",
            "num_mapped_edges",
            "num_gt_components",
            "num_pred_components",
        ):
            erl_payload[key][r_key] = int(erl_counts[key])

        path_payload["score"][r_key] = {}
        path_payload["connected"][r_key] = {}
        path_payload["total"][r_key] = {}
        for d_key in path_distance_keys:
            pc_counts = counts["path_connectivity"][d_key]
            connected = int(pc_counts["connected"])
            total = int(pc_counts["total"])
            path_payload["connected"][r_key][d_key] = connected
            path_payload["total"][r_key][d_key] = total
            path_payload["score"][r_key][d_key] = (
                float(connected / total) if total > 0 else float("nan")
            )

    return {
        "r_tolerant_erl": erl_payload,
        "r_tolerant_path_connectivity": path_payload,
    }


def _gt_segments_from_target(target: dict):
    """Build list of GT segments with neurite ids."""
    annotations = target.get("annotations", {})
    neurite_dicts = annotations.get("gt_neurites", [])
    segments = []
    for neurite_dict in neurite_dicts:
        if not isinstance(neurite_dict, dict):
            continue
        for neurite_id, pts in neurite_dict.items():
            arr = np.asarray(pts)
            if arr.shape[0] < 2:
                continue
            for p0, p1 in zip(arr[:-1], arr[1:]):
                segments.append(
                    (int(neurite_id), np.asarray(p0, dtype=float), np.asarray(p1, dtype=float))
                )
    return segments


def _pred_centers_to_vox(pred_boxes, scale):
    """
    Convert pred box centers into voxel coordinates that match GT coordinates.

    Heuristic:
      - If pred centers are in [0, 1] (normalized), scale by inferred GT extent.
    """
    if pred_boxes.size == 0:
        return np.zeros((0, 3), dtype=float)

    centers = pred_boxes[:, :3].astype(float, copy=False)  # (cz, cy, cx)
    cmax = np.nanmax(centers) if centers.size else 0.0

    if cmax <= 1.5:  # normalized-ish
        centers = centers * scale
    return centers


def _as_numpy(value, dtype=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def _empty_point_pred():
    return {
        "centers": np.zeros((0, 3), dtype=float),
        "inst_ids": np.zeros((0,), dtype=int),
        "neuron_ids": np.zeros((0,), dtype=int),
    }


def _connected_component_ids(coords, volume_shape):
    coords = np.asarray(coords, dtype=int)
    if coords.size == 0:
        return np.zeros((0,), dtype=int)

    vol_shape = np.asarray(volume_shape, dtype=int)
    mask = np.zeros(tuple(vol_shape.tolist()), dtype=np.uint8)
    mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    comp = cc_label(mask, connectivity=mask.ndim).astype(np.int32)
    inst_ids = comp[coords[:, 0], coords[:, 1], coords[:, 2]].astype(int, copy=False) - 1
    inst_ids[inst_ids < 0] = 0
    return inst_ids


def _point_pred_from_coords(coords, volume_shape=None):
    pts = _as_point_coords(coords, dtype=float)
    if pts is None or pts.size == 0:
        return _empty_point_pred()

    pts = np.rint(pts).astype(int, copy=False)
    if volume_shape is not None:
        vol_shape = np.asarray(volume_shape, dtype=int)
        pts = np.clip(pts, [0, 0, 0], vol_shape - 1)
    pts = np.unique(pts, axis=0)
    if pts.size == 0:
        return _empty_point_pred()

    inst_ids = (
        _connected_component_ids(pts, volume_shape)
        if volume_shape is not None
        else np.zeros((pts.shape[0],), dtype=int)
    )
    return {
        "centers": pts.astype(float, copy=False),
        "inst_ids": inst_ids,
        "neuron_ids": np.full((pts.shape[0],), -1, dtype=int),
    }


def _build_overlap_consensus_mask(positive_votes, cover_votes):
    pos = np.asarray(positive_votes, dtype=np.int32)
    cov = np.asarray(cover_votes, dtype=np.int32)
    if pos.shape != cov.shape:
        raise ValueError(f"Vote tensors must have the same shape, got {pos.shape} and {cov.shape}")
    return (cov > 0) & (pos == cov)


def _mask_to_centerline_pred(mask):
    mask_bin = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask_bin)
    if coords.size == 0:
        return _empty_point_pred()

    semantic_pred = {
        "centers": coords.astype(float, copy=False),
        "inst_ids": np.zeros((coords.shape[0],), dtype=int),
        "neuron_ids": np.full((coords.shape[0],), -1, dtype=int),
    }
    _, centerline = skeletonize_pred(semantic_pred, mask_bin.shape, merge_all_instance=True,)
    if not isinstance(centerline, dict):
        return _empty_point_pred()
    return _point_pred_from_coords(centerline.get("centers"), volume_shape=mask_bin.shape)


def _upsample_centerline_pred(pred_ds, full_shape, output_patch_size):
    centers_ds = _as_point_coords(pred_ds.get("centers"), dtype=float)
    inst_ids_ds = _as_numpy(pred_ds.get("inst_ids"), dtype=int)
    if centers_ds is None or centers_ds.size == 0:
        return _empty_point_pred()

    if inst_ids_ds is None or inst_ids_ds.size != centers_ds.shape[0]:
        inst_ids_ds = np.zeros((centers_ds.shape[0],), dtype=int)

    scale = int(output_patch_size)
    if scale <= 1:
        return _point_pred_from_coords(centers_ds, volume_shape=full_shape)

    up_coords = []
    up_inst = []
    connect_step = max(1, int(np.ceil(float(scale) * np.sqrt(3.0))))
    for inst_new, inst_id in enumerate(np.unique(inst_ids_ds)):
        comp_coords = centers_ds[inst_ids_ds == inst_id]
        if comp_coords.size == 0:
            continue
        comp_pts = np.unique(np.rint(comp_coords).astype(int, copy=False), axis=0)
        simplified = skeleton_to_simplified_paths(
            comp_pts,
            connectivity=26,
            keep_largest_cc=False,
            spur_len=0,
            smooth_win=1,
            rdp_eps=0.0,
            max_step=max(1, scale),
        )
        polylines = [branch["polyline"] for branch in simplified["branches"]]
        polylines.extend(loop["polyline"] for loop in simplified["loops"])
        if not polylines:
            polylines = [comp_pts.astype(float, copy=False)]

        dense_parts = []
        for poly in polylines:
            poly = np.asarray(poly, dtype=float)
            if poly.size == 0:
                continue
            dense = connect_neurite_points(poly * float(scale), max_step=connect_step)
            if dense.size == 0:
                continue
            dense_parts.append(np.rint(dense).astype(int, copy=False))
        if not dense_parts:
            dense_parts = [
                np.rint(comp_pts.astype(float, copy=False) * float(scale)).astype(int, copy=False)
            ]

        comp_up = np.concatenate(dense_parts, axis=0)
        comp_up = np.clip(comp_up, [0, 0, 0], np.asarray(full_shape, dtype=int) - 1)
        comp_up = np.unique(comp_up, axis=0)
        if comp_up.size == 0:
            continue
        up_coords.append(comp_up)
        up_inst.append(np.full((comp_up.shape[0],), inst_new, dtype=int))

    if not up_coords:
        return _empty_point_pred()

    coords = np.concatenate(up_coords, axis=0)
    inst_ids = np.concatenate(up_inst, axis=0)
    return {
        "centers": coords.astype(float, copy=False),
        "inst_ids": inst_ids,
        "neuron_ids": np.full((coords.shape[0],), -1, dtype=int),
    }


def _build_whole_hm_centerline_pred(
    whole_pred_hm, positive_votes, cover_votes, hm_threshold, full_shape, output_patch_size,
):
    hm_ds = (
        np.asarray(whole_pred_hm, dtype=np.float32)
        if whole_pred_hm is not None
        else np.zeros((0, 0, 0), dtype=np.float32)
    )
    if positive_votes is not None and cover_votes is not None:
        hm_mask_ds = _build_overlap_consensus_mask(positive_votes, cover_votes)
    else:
        hm_mask_ds = hm_ds > float(hm_threshold)
    centerline_ds = _mask_to_centerline_pred(hm_mask_ds)
    centerline_full = _upsample_centerline_pred(
        centerline_ds, full_shape=full_shape, output_patch_size=output_patch_size
    )
    return centerline_full, hm_mask_ds


def _normalize_pred_inputs(pred: dict, volume_shape=None):
    boxes_np = _as_numpy(pred.get("boxes"), dtype=float)
    if boxes_np is None or boxes_np.size == 0:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    if volume_shape is not None:
        scale = np.asarray(volume_shape, dtype=float)
        centers = _pred_centers_to_vox(boxes_np, scale)
    else:
        centers = boxes_np[:, :3].astype(float, copy=False)

    pred_inst = _as_numpy(pred.get("instances"), dtype=int)
    if pred_inst is None or pred_inst.size == 0:
        pred_inst = np.arange(centers.shape[0], dtype=int)

    pred_neur = _as_numpy(pred.get("neurons"), dtype=int)
    if pred_neur is None or pred_neur.size == 0:
        pred_neur = np.full(centers.shape[0], -1, dtype=int)

    print(f"Original pred inputs: {centers.shape[0]} centers")

    print(
        f"Normalized pred inputs: {centers.shape[0]} centers, instance shape {pred_inst.shape}, neuron shape {pred_neur.shape}"
    )
    return centers, pred_inst, pred_neur


def esa(
    pred: dict,
    target: dict,
    distance_threshold: float,
    exclude_nuclei: float = 0,
    coord_scale: float = 1.0,
    downsample_to_grid: bool = False,
):
    """
    Standard ESA (Peng/Vaa3D-style): average nearest-node distance, bidirectional.
    Also returns within-threshold rates as auxiliary (NOT part of ESA).
    SSD: average nearest-node distance after excluding distances <= distance_threshold.
    """
    tp_fp = []  # backward-compat tuple list of (pred_coords, matched_gt_or_none)
    fn = []  # backward-compat tuple list of (None, fn_gt_coords)

    distance_threshold = float(distance_threshold)
    gt_points, gt_inst, gt_neur = extract_gt_points(
        target, coord_scale=coord_scale, downsample_to_grid=downsample_to_grid,
    )
    pred_centers = _as_point_coords(pred.get("centers"), dtype=float)
    pred_inst = _as_numpy(pred.get("inst_ids"), dtype=int)
    pred_neur = _as_numpy(pred.get("neuron_ids"), dtype=int)
    if pred_centers is None or pred_centers.size == 0:
        pred_centers = np.zeros((0, 3), dtype=float)
    elif pred_centers.ndim == 1:
        pred_centers = pred_centers.reshape(1, -1)
    if pred_centers.shape[1] > 3:
        pred_centers = pred_centers[:, :3]
    if pred_inst is None or pred_inst.size != pred_centers.shape[0]:
        pred_inst = np.full((pred_centers.shape[0],), -1, dtype=int)
    if pred_neur is None or pred_neur.size != pred_centers.shape[0]:
        pred_neur = np.full((pred_centers.shape[0],), -1, dtype=int)
    if downsample_to_grid and pred_centers.size > 0:
        pred_centers = _downsample_points_to_grid(pred_centers, coord_scale=coord_scale)
        pred_centers, pred_inst, pred_neur = _dedupe_labeled_points(
            pred_centers, pred_inst, pred_neur
        )

    exclude_nuclei = float(exclude_nuclei)
    if exclude_nuclei > 0 and pred_centers.size > 0:
        exclusion_centers = _get_exclusion_centers(target)
        if downsample_to_grid and exclusion_centers is not None and exclusion_centers.size > 0:
            exclusion_centers = _downsample_points_to_grid(
                exclusion_centers, coord_scale=coord_scale
            )
            exclusion_centers = np.unique(exclusion_centers, axis=0)
        if exclusion_centers is not None and exclusion_centers.size > 0:
            if exclusion_centers.ndim == 1:
                exclusion_centers = exclusion_centers.reshape(1, -1)
            if exclusion_centers.shape[1] == 3:
                if gt_points is not None and gt_points.size > 0:
                    base_gt_tree = cKDTree(gt_points)
                    base_d_p2g = base_gt_tree.query(pred_centers, k=1)[0]
                    fp_candidates = base_d_p2g > distance_threshold
                else:
                    fp_candidates = np.ones(pred_centers.shape[0], dtype=bool)
                if fp_candidates.any():
                    fp_near_centers = ~_points_outside_radius_mask(
                        pred_centers[fp_candidates], exclusion_centers, exclude_nuclei,
                    )
                    if fp_near_centers.any():
                        pred_keep = np.ones(pred_centers.shape[0], dtype=bool)
                        pred_keep[np.flatnonzero(fp_candidates)[fp_near_centers]] = False
                        pred_centers = pred_centers[pred_keep]
                        pred_inst = pred_inst[pred_keep]
                        pred_neur = pred_neur[pred_keep]
    num_pred = int(pred_centers.shape[0])
    num_gt = int(gt_points.shape[0])

    d_p2g = None
    d_g2p = None
    nn_g = None
    nn_p = None
    tp_pred_mask = np.zeros(num_pred, dtype=bool)
    fp_pred_mask = np.zeros(num_pred, dtype=bool)
    tp_gt_mask = np.zeros(num_gt, dtype=bool)
    tp_points = np.zeros((0, 3), dtype=float)
    fp_points = np.zeros((0, 3), dtype=float)
    fn_points = np.zeros((0, 3), dtype=float)
    if num_pred > 0 and num_gt > 0:
        gt_tree = cKDTree(gt_points)
        pred_tree = cKDTree(pred_centers)
        d_p2g, nn_g = gt_tree.query(pred_centers, k=1)
        d_g2p, nn_p = pred_tree.query(gt_points, k=1)
        tp_pred_mask = d_p2g <= distance_threshold
        fp_pred_mask = d_p2g > distance_threshold
        tp_gt_mask = d_g2p <= distance_threshold
        tp_points = pred_centers[tp_pred_mask]
        fp_points = pred_centers[fp_pred_mask]
        fn_points = gt_points[~tp_gt_mask]
        tp_fp = [
            (pred_centers[i], gt_points[nn_g[i]]) if tp_pred_mask[i] else (pred_centers[i], None)
            for i in range(num_pred)
        ]
        fn = [(None, gt_points[i]) for i in np.flatnonzero(~tp_gt_mask)]
    elif num_pred > 0:
        tp_pred_mask = np.zeros(num_pred, dtype=bool)
        fp_pred_mask = np.ones(num_pred, dtype=bool)
        tp_gt_mask = np.zeros(0, dtype=bool)
        tp_points = np.zeros((0, 3), dtype=float)
        fp_points = pred_centers
        fn_points = np.zeros((0, 3), dtype=float)
        tp_fp = [(pred_centers[i], None) for i in range(num_pred)]
        fn = []
    elif num_gt > 0:
        tp_pred_mask = np.zeros(0, dtype=bool)
        fp_pred_mask = np.zeros(0, dtype=bool)
        tp_gt_mask = np.zeros(num_gt, dtype=bool)
        tp_points = np.zeros((0, 3), dtype=float)
        fp_points = np.zeros((0, 3), dtype=float)
        fn_points = gt_points
        tp_fp = []
        fn = [(None, gt_points[i]) for i in range(num_gt)]
    else:
        tp_pred_mask = np.zeros(0, dtype=bool)
        fp_pred_mask = np.zeros(0, dtype=bool)
        tp_gt_mask = np.zeros(0, dtype=bool)
        tp_points = np.zeros((0, 3), dtype=float)
        fp_points = np.zeros((0, 3), dtype=float)
        fn_points = np.zeros((0, 3), dtype=float)
        tp_fp = []
        fn = []

    # distance lists
    pred_to_gt_dists = []
    gt_to_pred_dists = []

    if num_pred > 0 and num_gt > 0:
        pred_to_gt_dists = d_p2g.tolist()
        gt_to_pred_dists = d_g2p.tolist()

        esa_p2g = float(np.mean(d_p2g))
        esa_g2p = float(np.mean(d_g2p))
        esa_mean = 0.5 * (esa_p2g + esa_g2p)

        far_p2g = d_p2g[d_p2g > distance_threshold]
        far_g2p = d_g2p[d_g2p > distance_threshold]
        ssd_p2g = float(np.mean(far_p2g)) if far_p2g.size > 0 else float("nan")
        ssd_g2p = float(np.mean(far_g2p)) if far_g2p.size > 0 else float("nan")
        ssd_mean = (
            0.5 * (ssd_p2g + ssd_g2p)
            if np.isfinite(ssd_p2g) and np.isfinite(ssd_g2p)
            else float("nan")
        )

        # auxiliary tolerance rates (optional to report)
        pred_within_r = float(np.mean(d_p2g <= distance_threshold))
        gt_within_r = float(np.mean(d_g2p <= distance_threshold))

        # optional: label transfer for your ARI bookkeeping
        gt_labels_for_preds_inst = gt_inst[nn_g].astype(int).tolist()
        gt_labels_for_preds_neur = gt_neur[nn_g].astype(int).tolist()

    else:
        # degenerate cases
        esa_p2g = float("inf") if (num_pred > 0 and num_gt == 0) else float("nan")
        esa_g2p = float("inf") if (num_gt > 0 and num_pred == 0) else float("nan")
        esa_mean = (
            float("inf")
            if (num_pred > 0 and num_gt == 0) or (num_gt > 0 and num_pred == 0)
            else float("nan")
        )

        ssd_p2g = float("inf") if (num_pred > 0 and num_gt == 0) else float("nan")
        ssd_g2p = float("inf") if (num_gt > 0 and num_pred == 0) else float("nan")
        ssd_mean = (
            float("inf")
            if (num_pred > 0 and num_gt == 0) or (num_gt > 0 and num_pred == 0)
            else float("nan")
        )

        pred_within_r = 0.0 if num_pred > 0 else float("nan")
        gt_within_r = 0.0 if num_gt > 0 else float("nan")

        gt_labels_for_preds_inst = [-1] * num_pred
        gt_labels_for_preds_neur = [-1] * num_pred
        pred_to_gt_dists = [float("inf")] * num_pred if num_pred > 0 and num_gt == 0 else []
        gt_to_pred_dists = [float("inf")] * num_gt if num_gt > 0 and num_pred == 0 else []

    return {
        # “ESA” proper (distance scores)
        "esa_p2g": esa_p2g,
        "esa_g2p": esa_g2p,
        "esa_mean": esa_mean,
        "ssd_p2g": ssd_p2g,
        "ssd_g2p": ssd_g2p,
        "ssd_mean": ssd_mean,
        # auxiliary tolerance rates (optional)
        "pred_within_r": pred_within_r,
        "gt_within_r": gt_within_r,
        # keep your existing bookkeeping fields
        "num_gt": num_gt,
        "num_pred": num_pred,
        "gt_points": gt_points,
        "pred_to_gt_dists": pred_to_gt_dists,
        "gt_to_pred_dists": gt_to_pred_dists,
        "pred_labels": {"instance": pred_inst.tolist(), "neuron": pred_neur.tolist()},
        "gt_labels_for_preds": {
            "instance": gt_labels_for_preds_inst,
            "neuron": gt_labels_for_preds_neur,
        },
        # direct ESA TP/FP/FN point outputs
        "tp_points": tp_points,
        "fp_points": fp_points,
        "fn_points": fn_points,
        "tp_pred_mask": tp_pred_mask,
        "fp_pred_mask": fp_pred_mask,
        "tp_gt_mask": tp_gt_mask,
        # backward compatibility
        "pred_to_gt": tp_fp,
        "undetected_gt": fn,
    }


def evaluate_instance_matching_metrics_merge_and_split(
    pred_dict: dict,
    gt_dict: dict,
    r: float = 2.0,
    cost_thr: float = None,
    tau_p: float = 0.7,
    tau_g: float = 0.7,
    big_cost: float = 1e9,
    tau_base: float = 0.2,
) -> dict:
    """
    Compute 3 metrics:
      1) one_to_one_pr: Hungarian 1-1 match, gated by symmetric cost + bidir coverage
      2) merge_allowed: Hungarian 1-1 match, gated by cost + (cov_g >= tau_g & cov_p >= tau_base)
      3) split_allowed: Hungarian 1-1 match, gated by cost + (cov_p >= tau_p & cov_g >= tau_base)

    Inputs:
      pred_dict: {pred_id: (Np,3) densified points}
      gt_dict:   {gt_id:   (Ng,3) densified points}

    Parameters:
      r: radius used for coverage (within-r)
      cost_thr: threshold on symmetric cost. If None, uses r.
      tau_p: min pred coverage cov_p to accept a pair
      tau_g: min gt coverage cov_g to accept a pair
      big_cost: sentinel cost for disallowed edges
      tau_base: base threshold for coverage
    """
    r = float(r)
    cost_thr = float(r if cost_thr is None else cost_thr)
    tau_p = float(tau_p)
    tau_g = float(tau_g)
    tau_base = float(tau_base)

    pred_ids = sorted(int(k) for k in pred_dict.keys())
    gt_ids = sorted(int(k) for k in gt_dict.keys())
    P, G = len(pred_ids), len(gt_ids)

    pred_pts = [_as_pts(pred_dict[i]) for i in pred_ids]
    gt_pts = [_as_pts(gt_dict[j]) for j in gt_ids]

    pred_trees = [cKDTree(pts) if pts.shape[0] > 0 else None for pts in pred_pts]
    gt_trees = [cKDTree(pts) if pts.shape[0] > 0 else None for pts in gt_pts]

    d_p2g = np.full((P, G), np.inf, dtype=np.float64)
    d_g2p = np.full((P, G), np.inf, dtype=np.float64)
    cov_p = np.zeros((P, G), dtype=np.float64)
    cov_g = np.zeros((P, G), dtype=np.float64)
    cost = np.full((P, G), np.inf, dtype=np.float64)

    # Compute pairwise stats
    for pi in range(P):
        Pi = pred_pts[pi]
        tp_tree = pred_trees[pi]
        if tp_tree is None or Pi.shape[0] == 0:
            continue
        for gj in range(G):
            Gj = gt_pts[gj]
            tg_tree = gt_trees[gj]
            if tg_tree is None or Gj.shape[0] == 0:
                continue

            # pred -> gt
            dp, _ = tg_tree.query(Pi, k=1)
            d_p2g[pi, gj] = float(np.mean(dp)) if dp.size else float("inf")
            cov_p[pi, gj] = float(np.mean(dp <= r)) if dp.size else 0.0

            # gt -> pred
            dg, _ = tp_tree.query(Gj, k=1)
            d_g2p[pi, gj] = float(np.mean(dg)) if dg.size else float("inf")
            cov_g[pi, gj] = float(np.mean(dg <= r)) if dg.size else 0.0

            cost[pi, gj] = 0.5 * (d_p2g[pi, gj] + d_g2p[pi, gj])

    eligible_1to1 = np.isfinite(cost) & ((cost <= cost_thr) | ((cov_p >= tau_p) & (cov_g >= tau_g)))
    eligible_merge = np.isfinite(cost) & (
        (cost <= cost_thr) | ((cov_g >= tau_g) & (cov_p >= tau_base))
    )
    eligible_split = np.isfinite(cost) & (
        (cost <= cost_thr) | ((cov_p >= tau_p) & (cov_g >= tau_base))
    )

    def _hungarian_match(eligible_mask: np.ndarray):
        cost_masked = np.where(eligible_mask, cost, big_cost)
        pred_to_gt = {}
        gt_to_pred = {}
        if P > 0 and G > 0:
            row_ind, col_ind = linear_sum_assignment(cost_masked)
            for rr, cc in zip(row_ind, col_ind):
                if cost_masked[rr, cc] >= big_cost:
                    continue
                pid = pred_ids[int(rr)]
                gid = gt_ids[int(cc)]
                pred_to_gt[pid] = gid
                gt_to_pred[gid] = pid
        return pred_to_gt, gt_to_pred, cost_masked

    # -------------------------
    # Metric 1: one-to-one PR
    # -------------------------
    pred_to_gt_1to1, gt_to_pred_1to1, cost_1to1 = _hungarian_match(eligible_1to1)

    tp1 = len(pred_to_gt_1to1)
    fp1 = P - tp1
    fn1 = G - tp1
    p1, r1, f11 = _prf(tp1, fp1, fn1)

    unmatched_pred_1to1 = [pid for pid in pred_ids if pid not in pred_to_gt_1to1]
    unmatched_gt_1to1 = [gid for gid in gt_ids if gid not in gt_to_pred_1to1]

    # shared match_id for viz
    matched_pairs_sorted = sorted(
        [(pid, gid) for pid, gid in pred_to_gt_1to1.items()], key=lambda x: (x[1], x[0])
    )
    match_id_by_pred = {}
    match_id_by_gt = {}
    for mid, (pid, gid) in enumerate(matched_pairs_sorted):
        match_id_by_pred[pid] = mid
        match_id_by_gt[gid] = mid

    one_to_one_pr = {
        "tp": int(tp1),
        "fp": int(fp1),
        "fn": int(fn1),
        "precision": p1,
        "recall": r1,
        "f1": f11,
        "pred_to_gt": pred_to_gt_1to1,
        "gt_to_pred": gt_to_pred_1to1,
        "unmatched_pred": unmatched_pred_1to1,
        "unmatched_gt": unmatched_gt_1to1,
        "match_id_by_pred": match_id_by_pred,
        "match_id_by_gt": match_id_by_gt,
    }

    # -----------------------------------------
    # Metric 2: merge allowed (1-to-1 match)
    # -----------------------------------------
    pred_to_gt_merge, gt_to_pred_merge, _ = _hungarian_match(eligible_merge)

    matched_gt = set(gt_to_pred_merge.keys())
    unmatched_gt_merge = [gid for gid in gt_ids if gid not in matched_gt]

    preds_with_match = sorted(pred_to_gt_merge.keys())
    preds_without_match = [pid for pid in pred_ids if pid not in pred_to_gt_merge]

    tp2 = len(pred_to_gt_merge)
    fp2 = P - tp2
    fn2 = G - tp2
    p2, r2, f12 = _prf(tp2, fp2, fn2)

    merge_allowed = {
        "tp": int(tp2),
        "fp": int(fp2),
        "fn": int(fn2),
        "precision": p2,
        "recall": r2,
        "f1": f12,
        "gt_to_pred": gt_to_pred_merge,
        "unmatched_gt": unmatched_gt_merge,
        "preds_with_match": preds_with_match,
        "preds_without_match": preds_without_match,
    }

    # -----------------------------------------
    # Metric 3: split allowed (1-to-1 match)
    # -----------------------------------------
    pred_to_gt_split, gt_to_pred_split, _ = _hungarian_match(eligible_split)

    matched_pred = set(pred_to_gt_split.keys())
    unmatched_pred_split = [pid for pid in pred_ids if pid not in matched_pred]

    gts_covered = sorted(set(pred_to_gt_split.values()))
    gts_uncovered = [gid for gid in gt_ids if gid not in set(gts_covered)]

    tp3 = len(pred_to_gt_split)
    fp3 = P - tp3
    fn3 = G - tp3
    p3, r3, f13 = _prf(tp3, fp3, fn3)

    split_allowed = {
        "tp": int(tp3),
        "fp": int(fp3),
        "fn": int(fn3),
        "precision": p3,
        "recall": r3,
        "f1": f13,
        "pred_to_gt": pred_to_gt_split,
        "unmatched_pred": unmatched_pred_split,
        "gts_covered": gts_covered,
        "gts_uncovered": gts_uncovered,
    }

    return {
        "pred_ids": pred_ids,
        "gt_ids": gt_ids,
        "pairwise": {
            "d_p2g": d_p2g,
            "d_g2p": d_g2p,
            "cov_p": cov_p,
            "cov_g": cov_g,
            "cost": cost,
            "eligible": eligible_1to1,
            "eligible_1to1": eligible_1to1,
            "eligible_merge": eligible_merge,
            "eligible_split": eligible_split,
            "r": r,
            "cost_thr": cost_thr,
            "tau_p": tau_p,
            "tau_g": tau_g,
        },
        "one_to_one_pr": one_to_one_pr,
        "merge_allowed": merge_allowed,
        "split_allowed": split_allowed,
    }


def viz_esa_distances(
    out_dir,
    voxel,
    target,
    pred,
    metrics_esa,
    pred_hm=None,
    output_patch_size=1,
    z_step=0,
    dilation=True,
    exclude_nuclei=0,
    subvol_viz_xy_size=192,
    subvol_viz_dirname="subvol_top_left_192x192",
    display_low_percentile=0.5,
    display_high_percentile=99.0,
):
    if isinstance(voxel, torch.Tensor):
        arr = voxel.detach().cpu().numpy()
    else:
        arr = np.asarray(voxel)
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        arr = arr.max(axis=0)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected volume shape for projection: {arr.shape}")

    hm_arr = None
    if pred_hm is not None:
        hm_arr = (
            pred_hm.detach().cpu().numpy()
            if isinstance(pred_hm, torch.Tensor)
            else np.asarray(pred_hm)
        )
        hm_arr = np.squeeze(hm_arr)
        if hm_arr.ndim == 4:
            hm_arr = hm_arr.max(axis=0)
        if hm_arr.ndim != 3:
            raise ValueError(f"Unexpected heatmap shape for projection: {hm_arr.shape}")

    segments = _gt_segments_from_target(target)
    pred_centers = _as_numpy(pred.get("centers"), dtype=float)
    pred_inst_ids = _as_numpy(pred.get("inst_ids"))
    annotations = target.get("annotations", {})
    gt_centers = _flatten_center_sets(annotations.get("cell_centers"))
    exclusion_centers = _get_exclusion_centers(target)
    subvol_viz_xy_size = int(subvol_viz_xy_size or 0)
    output_patch_size = max(int(output_patch_size), 1)
    display_low_percentile = float(display_low_percentile)
    display_high_percentile = float(display_high_percentile)

    def _to_points3d(points):
        if points is None:
            return np.zeros((0, 3), dtype=float)
        arr = (
            points.detach().cpu().numpy()
            if torch.is_tensor(points)
            else np.asarray(points, dtype=float)
        )
        if arr.size == 0:
            return np.zeros((0, 3), dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            return np.zeros((0, 3), dtype=float)
        return arr[:, :3].astype(float, copy=False)

    def _crop_points_xy(points, y0, y1, x0, x1):
        arr = _to_points3d(points)
        if arr.size == 0:
            return arr
        mask = (arr[:, 1] >= y0) & (arr[:, 1] < y1) & (arr[:, 2] >= x0) & (arr[:, 2] < x1)
        arr = arr[mask].copy()
        if arr.size:
            arr[:, 1] -= y0
            arr[:, 2] -= x0
        return arr

    def _crop_centers_xy(centers, inst_ids, y0, y1, x0, x1):
        if centers is None or inst_ids is None or centers.size == 0:
            return centers, inst_ids
        mask = (
            (centers[:, 1] >= y0)
            & (centers[:, 1] < y1)
            & (centers[:, 2] >= x0)
            & (centers[:, 2] < x1)
        )
        centers_crop = centers[mask].copy()
        if centers_crop.size:
            centers_crop[:, 1] -= y0
            centers_crop[:, 2] -= x0
        return centers_crop, inst_ids[mask]

    def _crop_segments_xy(all_segments, y0, y1, x0, x1):
        cropped = []
        for neurite_id, p0, p1 in all_segments:
            if max(float(p0[1]), float(p1[1])) < y0 or min(float(p0[1]), float(p1[1])) >= y1:
                continue
            if max(float(p0[2]), float(p1[2])) < x0 or min(float(p0[2]), float(p1[2])) >= x1:
                continue
            p0_crop = np.asarray(p0, dtype=float).copy()
            p1_crop = np.asarray(p1, dtype=float).copy()
            p0_crop[1] -= y0
            p1_crop[1] -= y0
            p0_crop[2] -= x0
            p1_crop[2] -= x0
            cropped.append((neurite_id, p0_crop, p1_crop))
        return cropped

    def _crop_metrics_xy(metrics, y0, y1, x0, x1):
        if metrics is None:
            metrics = {}
        return {
            "tp_points": _crop_points_xy(metrics.get("tp_points"), y0, y1, x0, x1),
            "fp_points": _crop_points_xy(metrics.get("fp_points"), y0, y1, x0, x1),
            "fn_points": _crop_points_xy(metrics.get("fn_points"), y0, y1, x0, x1),
            "gt_points": _crop_points_xy(metrics.get("gt_points"), y0, y1, x0, x1),
        }

    def _crop_hm_projection_xy(hm_proj, y0, y1, x0, x1):
        if hm_proj is None:
            return None
        hm_y0 = max(0, y0 // output_patch_size)
        hm_x0 = max(0, x0 // output_patch_size)
        hm_y1 = min(hm_proj.shape[0], int(np.ceil(y1 / output_patch_size)))
        hm_x1 = min(hm_proj.shape[1], int(np.ceil(x1 / output_patch_size)))
        return hm_proj[hm_y0:hm_y1, hm_x0:hm_x1]

    def _robust_limits(values, low, high):
        lo, hi = np.percentile(values, [low, high])
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo):
            lo = float(np.nanmin(values))
        if not np.isfinite(hi):
            hi = float(np.nanmax(values))
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _robust_normalize(values, low, high):
        norm = (values.astype(np.float32, copy=False) - float(low)) / (float(high) - float(low))
        return np.clip(norm, 0.0, 1.0)

    def _robust_max_projection(volume):
        mip = np.asarray(volume).max(axis=0).astype(np.float32, copy=False)
        low, high = _robust_limits(
            mip.reshape(-1), low=display_low_percentile, high=display_high_percentile,
        )
        return _robust_normalize(mip, low, high)

    def _filter_segments(all_segments, start_z, end_z):
        if start_z == 0 and end_z == arr.shape[0]:
            return all_segments
        filtered = []
        for neurite_id, p0, p1 in all_segments:
            z0 = float(p0[0])
            z1 = float(p1[0])
            if max(z0, z1) >= start_z and min(z0, z1) < end_z:
                filtered.append((neurite_id, p0, p1))
        return filtered

    def _filter_centers(centers, inst_ids, start_z, end_z):
        if centers is None or inst_ids is None or centers.size == 0:
            return centers, inst_ids
        mask = (centers[:, 0] >= start_z) & (centers[:, 0] < end_z)
        return centers[mask], inst_ids[mask]

    def _filter_gt_centers(centers, start_z, end_z):
        if centers is None or centers.size == 0:
            return centers
        mask = (centers[:, 0] >= start_z) & (centers[:, 0] < end_z)
        return centers[mask]

    def _filter_metrics(metrics, start_z, end_z):
        filtered = {
            "tp_points": np.zeros((0, 3), dtype=float),
            "fp_points": np.zeros((0, 3), dtype=float),
            "fn_points": np.zeros((0, 3), dtype=float),
            "gt_points": np.zeros((0, 3), dtype=float),
        }

        if metrics is None:
            return filtered

        tp_points = _to_points3d(metrics.get("tp_points"))
        fp_points = _to_points3d(metrics.get("fp_points"))
        fn_points = _to_points3d(metrics.get("fn_points"))

        # Backward compatibility: reconstruct TP/FP/FN points from older tuple fields.
        if tp_points.size == 0 and fp_points.size == 0 and fn_points.size == 0:
            legacy_tp = []
            legacy_fp = []
            legacy_fn = []
            for pred_coord, gt_coord in metrics.get("pred_to_gt", []):
                if pred_coord is None:
                    continue
                if gt_coord is None:
                    legacy_fp.append(pred_coord)
                else:
                    legacy_tp.append(pred_coord)
            for _, gt_coord in metrics.get("undetected_gt", []):
                if gt_coord is not None:
                    legacy_fn.append(gt_coord)
            tp_points = _to_points3d(legacy_tp)
            fp_points = _to_points3d(legacy_fp)
            fn_points = _to_points3d(legacy_fn)

        if tp_points.size:
            tp_mask = (tp_points[:, 0] >= start_z) & (tp_points[:, 0] < end_z)
            filtered["tp_points"] = tp_points[tp_mask]
        if fp_points.size:
            fp_mask = (fp_points[:, 0] >= start_z) & (fp_points[:, 0] < end_z)
            filtered["fp_points"] = fp_points[fp_mask]
        if fn_points.size:
            fn_mask = (fn_points[:, 0] >= start_z) & (fn_points[:, 0] < end_z)
            filtered["fn_points"] = fn_points[fn_mask]

        gt_points = metrics.get("gt_points")
        if gt_points is not None and np.asarray(gt_points).size:
            gt_points = np.asarray(gt_points, dtype=float)
            if gt_points.ndim == 1:
                gt_points = gt_points.reshape(1, -1)
            gt_mask = (gt_points[:, 0] >= start_z) & (gt_points[:, 0] < end_z)
            filtered["gt_points"] = gt_points[gt_mask]
        return filtered

    def _render(max_proj, segs, centers, inst_ids, chunk_metrics, hm_proj, out_path, gt_centers):
        def _dilate_rgb(rgb, iterations=1):
            if not dilation:
                return rgb
            if rgb is None or rgb.size == 0:
                return rgb
            h, w, _ = rgb.shape
            flat = rgb.reshape(-1, 3)
            unique_colors = np.unique(flat, axis=0)
            struct = np.ones((3, 3), dtype=bool)
            out = rgb.copy()
            for color in unique_colors:
                if np.allclose(color, 0.0):
                    continue
                mask = np.all(rgb == color, axis=-1)
                if not mask.any():
                    continue
                dilated = binary_dilation(mask, structure=struct, iterations=iterations)
                out[dilated] = color
            return out

        max_proj = max_proj.astype(float)

        gt_ids = sorted({neurite_id for neurite_id, _, _ in segs})
        cmap = plt.get_cmap("tab20")
        gt_colors = {inst_id: cmap(i % cmap.N) for i, inst_id in enumerate(gt_ids)}
        inst_ids = inst_ids if inst_ids is not None else np.asarray([])
        pred_unique_ids = np.unique(inst_ids) if inst_ids.size else []
        pred_colors = {inst_id: cmap(i % cmap.N) for i, inst_id in enumerate(pred_unique_ids)}

        num_subplots = 5 if hm_proj is not None else 4
        fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
        seg_ax = axes[1]
        esa_ax = axes[2]
        instance_ax = axes[3]
        hm_ax = None
        for i, ax in enumerate(axes):
            if i in [0, 1]:
                ax.imshow(max_proj, cmap="gray", interpolation="nearest")
            ax.set_axis_off()

        if hm_proj is not None:
            hm_ax = axes[4]
            hm_ax.imshow(hm_proj, cmap="magma")

        for neurite_id, p0, p1 in segs:
            color = gt_colors.get(neurite_id, cmap(0))
            seg_ax.plot([p0[2], p1[2]], [p0[1], p1[1]], color=color, linewidth=2.0, alpha=0.8)

        tp_coords = _to_points3d(chunk_metrics.get("tp_points"))
        fp_coords = _to_points3d(chunk_metrics.get("fp_points"))
        fn_coords = _to_points3d(chunk_metrics.get("fn_points"))
        if exclude_nuclei > 0 and fp_coords.size > 0 and exclusion_centers.size > 0:
            fp_keep_mask = _points_outside_radius_mask(fp_coords, exclusion_centers, exclude_nuclei)
            excluded_fp_coords = fp_coords[~fp_keep_mask]
            fp_coords = fp_coords[fp_keep_mask]
            if centers is not None and inst_ids is not None and excluded_fp_coords.size > 0:
                center_arr = _as_point_coords(centers, dtype=float)
                inst_arr = _as_numpy(inst_ids)
                if (
                    center_arr is not None
                    and inst_arr is not None
                    and inst_arr.size == center_arr.shape[0]
                ):
                    excluded_tree = cKDTree(excluded_fp_coords)
                    keep_centers = excluded_tree.query(center_arr, k=1)[0] > 1e-6
                    if not keep_centers.all():
                        centers = center_arr[keep_centers]
                        inst_ids = inst_arr[keep_centers]

        # 3th subplot: point-wise matching visualization (raw image scale)
        h, w = max_proj.shape[0], max_proj.shape[1]

        def _coords_to_mask(coords):
            mask = np.zeros((h, w), dtype=bool)
            if coords is None:
                return mask
            arr = np.asarray(coords, dtype=float)
            if arr.size == 0:
                return mask
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            y = np.clip(arr[:, 1].round().astype(int), 0, h - 1)
            x = np.clip(arr[:, 2].round().astype(int), 0, w - 1)
            mask[y, x] = True
            return mask

        tp_seed = _coords_to_mask(tp_coords)
        fp_seed = _coords_to_mask(fp_coords)
        fn_seed = _coords_to_mask(fn_coords)

        # If the same projected pixel appears in multiple groups, force TP priority.
        fp_seed = fp_seed & (~tp_seed)
        fn_seed = fn_seed & (~tp_seed) & (~fp_seed)

        tp_mask = tp_seed
        fp_mask = fp_seed
        fn_mask = fn_seed
        if dilation:
            struct = np.ones((3, 3), dtype=bool)
            tp_mask = binary_dilation(tp_seed, structure=struct, iterations=1)
            fp_mask = binary_dilation(fp_seed, structure=struct, iterations=1)
            fn_mask = binary_dilation(fn_seed, structure=struct, iterations=1)

        # Priority: TP (green) > FP (red) > FN (cyan)
        fp_mask = fp_mask & (~tp_mask)
        fn_mask = fn_mask & (~tp_mask) & (~fp_mask)

        esa_rgb = np.zeros((h, w, 3), dtype=np.float32)
        esa_rgb[fp_mask] = (1.0, 0.0, 0.0)
        esa_rgb[fn_mask] = (0, 1.0, 1.0)
        esa_rgb[tp_mask] = (0.0, 0.5, 0.0)
        esa_ax.imshow(esa_rgb, interpolation="none", resample=False)

        instance_rgb = np.zeros((h, w, 3), dtype=np.float32)
        if centers is not None and inst_ids is not None and len(centers):
            if centers.ndim == 2 and centers.shape[1] >= 3:
                inst_y = np.clip(np.round(centers[:, 1]).astype(int), 0, h - 1)
                inst_x = np.clip(np.round(centers[:, 2]).astype(int), 0, w - 1)
                unique_ids, inv = np.unique(inst_ids, return_inverse=True)
                unique_colors = np.array(
                    [pred_colors.get(inst_id, cmap(0))[:3] for inst_id in unique_ids],
                    dtype=np.float32,
                )
                instance_rgb[inst_y, inst_x] = unique_colors[inv]
        instance_rgb = _dilate_rgb(instance_rgb, iterations=1)
        instance_ax.imshow(instance_rgb, interpolation="none", resample=False)

        for ax in axes[:4]:
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
        if hm_ax is not None and hm_proj is not None:
            hm_h, hm_w = hm_proj.shape[:2]
            hm_ax.set_xlim(-0.5, hm_w - 0.5)
            hm_ax.set_ylim(hm_h - 0.5, -0.5)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(pad=0.1)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _render_top_left_subvol(
        volume, segs, centers, inst_ids, chunk_metrics, hm_proj, out_path, gt_centers
    ):
        if subvol_viz_xy_size <= 0:
            return
        h, w = volume.shape[-2:]
        y0, x0 = 0, 0
        y1 = min(subvol_viz_xy_size, h)
        x1 = min(subvol_viz_xy_size, w)
        if y1 <= y0 or x1 <= x0:
            return

        max_proj_crop = _robust_max_projection(volume[:, y0:y1, x0:x1])
        hm_proj_crop = _crop_hm_projection_xy(hm_proj, y0, y1, x0, x1)
        segs_crop = _crop_segments_xy(segs, y0, y1, x0, x1)
        centers_crop, inst_ids_crop = _crop_centers_xy(centers, inst_ids, y0, y1, x0, x1)
        metrics_crop = _crop_metrics_xy(chunk_metrics, y0, y1, x0, x1)
        gt_centers_crop = _crop_points_xy(gt_centers, y0, y1, x0, x1)
        _render(
            max_proj_crop,
            segs_crop,
            centers_crop,
            inst_ids_crop,
            metrics_crop,
            hm_proj_crop,
            out_path,
            gt_centers_crop,
        )

    depth = arr.shape[0]
    if z_step and z_step > 0:
        out_dir_path = Path(out_dir)
        if out_dir_path.suffix:
            out_dir_path = out_dir_path.parent
        subvol_out_dir_path = out_dir_path / subvol_viz_dirname
        num_parts = (depth - 1) // z_step + 1
        print(
            f"Volume shape: {arr.shape}, Visualizing in {num_parts} parts with z_step={z_step}..."
        )
        for part_idx in range(num_parts):
            start_z = part_idx * z_step
            end_z = min(start_z + z_step, depth)
            chunk = arr[start_z:end_z]
            if chunk.size == 0:
                continue
            max_proj = _robust_max_projection(chunk)
            hm_proj = None
            if hm_arr is not None:
                hm_chunk = hm_arr[start_z // output_patch_size : end_z // output_patch_size]
                if hm_chunk.size:
                    hm_proj = hm_chunk.max(axis=0)
            chunk_segments = _filter_segments(segments, start_z, end_z)
            chunk_centers, chunk_inst_ids = _filter_centers(
                pred_centers, pred_inst_ids, start_z, end_z
            )
            chunk_gt_centers = _filter_gt_centers(gt_centers, start_z, end_z)
            chunk_metrics = _filter_metrics(metrics_esa, start_z, end_z)
            out_path = out_dir_path / f"viz_{start_z}.png"
            _render(
                max_proj,
                chunk_segments,
                chunk_centers,
                chunk_inst_ids,
                chunk_metrics,
                hm_proj,
                out_path,
                chunk_gt_centers,
            )
            subvol_out_path = subvol_out_dir_path / f"viz_{start_z}.png"
            _render_top_left_subvol(
                chunk,
                chunk_segments,
                chunk_centers,
                chunk_inst_ids,
                chunk_metrics,
                hm_proj,
                subvol_out_path,
                chunk_gt_centers,
            )
    else:
        max_proj = _robust_max_projection(arr)
        hm_proj = hm_arr.max(axis=0) if hm_arr is not None else None
        out_path = Path(out_dir)
        if out_path.suffix == "":
            out_path = out_path / "viz_0.png"
        _render(
            max_proj,
            segments,
            pred_centers,
            pred_inst_ids,
            metrics_esa,
            hm_proj,
            out_path,
            gt_centers,
        )
        subvol_out_path = out_path.parent / subvol_viz_dirname / out_path.name
        _render_top_left_subvol(
            arr,
            segments,
            pred_centers,
            pred_inst_ids,
            metrics_esa,
            hm_proj,
            subvol_out_path,
            gt_centers,
        )


def _to_np(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _group_by_instance(points_np, inst_np, mask_np=None):
    if points_np is None or points_np.size == 0:
        return {}
    if mask_np is None:
        mask_np = np.ones((points_np.shape[0],), dtype=bool)
    idx = np.where(mask_np)[0]
    if idx.size == 0:
        return {}
    grouped = defaultdict(list)
    for ii in idx:
        grouped[int(inst_np[ii])].append(points_np[ii])
    return {k: np.asarray(v, dtype=float) for k, v in grouped.items()}


def _soft_vote_keep_mask(curr_pts, prev_pts, r, k):
    nc = curr_pts.shape[0]
    if nc == 0:
        return np.zeros((0,), dtype=bool)
    if prev_pts is None or prev_pts.shape[0] == 0:
        return np.zeros((nc,), dtype=bool)
    tree = cKDTree(prev_pts)
    neigh = tree.query_ball_point(curr_pts, r)
    keep = np.fromiter((len(n) >= k for n in neigh), count=nc, dtype=bool)
    return keep


def _match_instances_by_overlap(prev_by_id, curr_by_id, r, ratio_thr):
    matches = {}
    if not prev_by_id or not curr_by_id:
        return matches
    prev_trees = {}
    for pid, pts in prev_by_id.items():
        if pts.shape[0] > 0:
            prev_trees[pid] = cKDTree(pts)
    for cid, cpts in curr_by_id.items():
        if cpts.shape[0] == 0:
            continue
        best_pid = None
        best_score = -1.0
        c_n = float(cpts.shape[0])
        for pid, tree in prev_trees.items():
            neigh = tree.query_ball_point(cpts, r)
            supported = sum(1 for n in neigh if len(n) > 0)
            score = supported / max(1.0, c_n)
            if score >= ratio_thr and score > best_score:
                best_score = score
                best_pid = pid
        if best_pid is not None:
            matches[int(cid)] = (int(best_pid), float(best_score))
    return matches


def overlap_filter_and_merge(
    boxes_tile_norm,
    labels,
    logits,
    inst_ids_tile,
    tile_origin_zyx,
    input_size_zyx,
    vol_size_zyx,
    overlap_zyx,
    instance_offset,
    uf_find,
    uf_union,
    tile_overlap_stash,
    tile_key,
    merge_dist=2.0,
    vote_k=2,
    ratio_thr=0.4,
    require_two_sides=False,
):
    n = int(boxes_tile_norm.shape[0])
    if n == 0:
        tile_overlap_stash[tile_key] = {
            "z_end": np.zeros((0, 3), dtype=float),
            "y_end": np.zeros((0, 3), dtype=float),
            "x_end": np.zeros((0, 3), dtype=float),
            "z_end_inst": np.zeros((0,), dtype=np.int64),
            "y_end_inst": np.zeros((0,), dtype=np.int64),
            "x_end_inst": np.zeros((0,), dtype=np.int64),
        }
        return boxes_tile_norm[:0], labels[:0], logits[:0], inst_ids_tile[:0], 0

    input_size_t = boxes_tile_norm.new_tensor(input_size_zyx)
    vol_size_t = boxes_tile_norm.new_tensor(vol_size_zyx)
    origin_t = boxes_tile_norm.new_tensor(tile_origin_zyx)

    centers_local = boxes_tile_norm[:, :3] * input_size_t
    centers_global = centers_local + origin_t
    in_vol = (centers_global >= 0).all(dim=1) & (centers_global < vol_size_t).all(dim=1)

    oz, oy, ox = overlap_zyx
    left_z, left_y, left_x = oz // 2, oy // 2, ox // 2
    right_z, right_y, right_x = oz - left_z, oy - left_y, ox - left_x
    valid_low = centers_local.new_tensor([left_z, left_y, left_x])
    valid_high = input_size_t - centers_local.new_tensor([right_z, right_y, right_x])
    interior = ((centers_local >= valid_low) & (centers_local < valid_high)).all(dim=1) & in_vol

    start_z = (centers_local[:, 0] >= 0) & (centers_local[:, 0] < oz) & in_vol
    start_y = (centers_local[:, 1] >= 0) & (centers_local[:, 1] < oy) & in_vol
    start_x = (centers_local[:, 2] >= 0) & (centers_local[:, 2] < ox) & in_vol

    end_z = (
        (centers_local[:, 0] >= (input_size_t[0] - oz))
        & (centers_local[:, 0] < input_size_t[0])
        & in_vol
    )
    end_y = (
        (centers_local[:, 1] >= (input_size_t[1] - oy))
        & (centers_local[:, 1] < input_size_t[1])
        & in_vol
    )
    end_x = (
        (centers_local[:, 2] >= (input_size_t[2] - ox))
        & (centers_local[:, 2] < input_size_t[2])
        & in_vol
    )

    inst_global = inst_ids_tile + int(instance_offset)

    centers_global_np = _to_np(centers_global)

    keep_vote = np.zeros((n,), dtype=bool)

    i, j, k = tile_key

    def _apply_vote(dir_name, prev_key, curr_mask_np):
        nonlocal keep_vote
        prev = tile_overlap_stash.get(prev_key, None)
        if prev is None:
            return
        prev_pts = prev.get(f"{dir_name}_end", None)
        if prev_pts is None or prev_pts.shape[0] == 0:
            return
        idx = np.where(curr_mask_np)[0]
        if idx.size == 0:
            return
        curr_pts = centers_global_np[idx]
        voted = _soft_vote_keep_mask(curr_pts, prev_pts, r=float(merge_dist), k=int(vote_k))
        keep_vote[idx] |= voted

    start_z_np = _to_np(start_z)
    start_y_np = _to_np(start_y)
    start_x_np = _to_np(start_x)

    _apply_vote("z", (i - 1, j, k), start_z_np)
    _apply_vote("y", (i, j - 1, k), start_y_np)
    _apply_vote("x", (i, j, k - 1), start_x_np)

    if require_two_sides:
        overlap_count = start_z_np.astype(int) + start_y_np.astype(int) + start_x_np.astype(int)
        support_count = np.zeros((n,), dtype=int)

        def _count_vote(dir_name, prev_key, curr_mask_np):
            nonlocal support_count
            prev = tile_overlap_stash.get(prev_key, None)
            if prev is None:
                return
            prev_pts = prev.get(f"{dir_name}_end", None)
            if prev_pts is None or prev_pts.shape[0] == 0:
                return
            idx = np.where(curr_mask_np)[0]
            if idx.size == 0:
                return
            curr_pts = centers_global_np[idx]
            voted = _soft_vote_keep_mask(curr_pts, prev_pts, r=float(merge_dist), k=int(vote_k))
            support_count[idx] += voted.astype(int)

        _count_vote("z", (i - 1, j, k), start_z_np)
        _count_vote("y", (i, j - 1, k), start_y_np)
        _count_vote("x", (i, j, k - 1), start_x_np)

        keep_vote = np.where(overlap_count >= 2, support_count >= 2, keep_vote)

    start_any = start_z_np | start_y_np | start_x_np
    interior_np = _to_np(interior)
    keep_np = interior_np | (start_any & keep_vote)

    removed_by_vote = int(np.sum(start_any & ~keep_vote & ~interior_np))

    keep_t = boxes_tile_norm.new_tensor(keep_np, dtype=torch.bool)
    if keep_t.sum().item() == 0:
        tile_overlap_stash[tile_key] = {
            "z_end": np.zeros((0, 3), dtype=float),
            "y_end": np.zeros((0, 3), dtype=float),
            "x_end": np.zeros((0, 3), dtype=float),
            "z_end_inst": np.zeros((0,), dtype=np.int64),
            "y_end_inst": np.zeros((0,), dtype=np.int64),
            "x_end_inst": np.zeros((0,), dtype=np.int64),
        }
        return boxes_tile_norm[:0], labels[:0], logits[:0], inst_global[:0], removed_by_vote

    centers_global_kept = centers_global[keep_t]
    sizes_global_norm = boxes_tile_norm[:, 3:6][keep_t] * (input_size_t / vol_size_t)
    boxes_global_norm = torch.cat((centers_global_kept / vol_size_t, sizes_global_norm), dim=1)

    labels_kept = labels[keep_t]
    logits_kept = logits[keep_t]
    inst_kept = inst_global[keep_t]

    centers_kept_np = _to_np(centers_global_kept)
    inst_kept_np = _to_np(inst_kept)

    start_z_kept = start_z_np[keep_np]
    start_y_kept = start_y_np[keep_np]
    start_x_kept = start_x_np[keep_np]

    curr_z = _group_by_instance(centers_kept_np, inst_kept_np, start_z_kept)
    curr_y = _group_by_instance(centers_kept_np, inst_kept_np, start_y_kept)
    curr_x = _group_by_instance(centers_kept_np, inst_kept_np, start_x_kept)

    def _merge_dir(dir_name, prev_key, curr_groups):
        prev = tile_overlap_stash.get(prev_key, None)
        if prev is None:
            return
        prev_pts = prev.get(f"{dir_name}_end", None)
        prev_inst = prev.get(f"{dir_name}_end_inst", None)
        if prev_pts is None or prev_inst is None or prev_pts.shape[0] == 0:
            return
        prev_groups = _group_by_instance(prev_pts, prev_inst, None)
        matches = _match_instances_by_overlap(
            prev_groups, curr_groups, r=float(merge_dist), ratio_thr=float(ratio_thr)
        )
        for cid, (pid, _score) in matches.items():
            uf_union(pid, cid)

    _merge_dir("z", (i - 1, j, k), curr_z)
    _merge_dir("y", (i, j - 1, k), curr_y)
    _merge_dir("x", (i, j, k - 1), curr_x)

    end_z_np = _to_np(end_z)
    end_y_np = _to_np(end_y)
    end_x_np = _to_np(end_x)

    end_z_kept = end_z_np[keep_np]
    end_y_kept = end_y_np[keep_np]
    end_x_kept = end_x_np[keep_np]

    tile_overlap_stash[tile_key] = {
        "z_end": centers_kept_np[end_z_kept]
        if np.any(end_z_kept)
        else np.zeros((0, 3), dtype=float),
        "y_end": centers_kept_np[end_y_kept]
        if np.any(end_y_kept)
        else np.zeros((0, 3), dtype=float),
        "x_end": centers_kept_np[end_x_kept]
        if np.any(end_x_kept)
        else np.zeros((0, 3), dtype=float),
        "z_end_inst": inst_kept_np[end_z_kept]
        if np.any(end_z_kept)
        else np.zeros((0,), dtype=np.int64),
        "y_end_inst": inst_kept_np[end_y_kept]
        if np.any(end_y_kept)
        else np.zeros((0,), dtype=np.int64),
        "x_end_inst": inst_kept_np[end_x_kept]
        if np.any(end_x_kept)
        else np.zeros((0,), dtype=np.int64),
    }

    return boxes_global_norm, labels_kept, logits_kept, inst_kept, removed_by_vote


def _load_models(args, device):
    count = len(args.models or [])
    if not count or len(args.model_archs or []) != count or len(args.plot_labels or []) != count:
        raise ValueError("Provide matching --models, --model_archs and --plot_labels lists.")
    checkpoints = args.checkpoints or ["checkpoint-399.pth"] * count
    if len(checkpoints) != count:
        raise ValueError("Provide one --checkpoints entry per model.")
    models = []
    for architecture, name, checkpoint in zip(args.model_archs, args.models, checkpoints):
        constructor = models_mae_axon_hm.__dict__.get(architecture)
        if not architecture.startswith("mae_vit_") or not callable(constructor):
            raise ValueError(f"Unsupported neurite architecture: {architecture!r}")
        path = Path(checkpoint)
        if not path.is_absolute():
            path = Path(args.ckpt_root) / f"brainmae-{args.encoder}-{name}" / path
        print(f"Loading {architecture} from {path}")
        model = constructor().to(device)
        model.load_state_dict(torch.load(path, map_location="cpu")["model"], strict=True)
        model.eval()
        models.append(model)
    return models


def _run_single_hm_threshold(args, hm_threshold, eval_radii, timestamp):
    args.hm_threshold = float(hm_threshold)
    hm_threshold_key = _radius_key(args.hm_threshold)
    path_connectivity_distances = [float(d) for d in args.path_connectivity_distances]
    device = torch.device(args.device)
    # selected_axon_id is for fMOST only
    data_path = Path(args.data_path)
    data_vols = load_annotations_from_json(
        args.testdata,
        data_path,
        labeled=args.labeled,
        selected_axon_id=args.selected_axon_id,
        fMOST=args.fMOST,
    )

    primary_eval_radius = 4  # was 2 for fmost
    print(f"Evaluating hm_threshold={hm_threshold_key}")
    print(f"Evaluation radii: {', '.join(_radius_key(r) for r in eval_radii)}")
    print(
        f"Path connectivity distances: {', '.join(_radius_key(d) for d in path_connectivity_distances)}"
    )

    models = _load_models(args, device)
    model_threshold_results = {}
    for model_name, model, plot_label, model_arch in zip(
        args.models, models, args.plot_labels, args.model_archs
    ):
        print(f"evaluating {model_name}...")
        summary_viz_dir, hm_eval_dir = _build_eval_dirs(
            args, model_name, args.hm_threshold, timestamp
        )
        use_whole_volume_hm_path = bool(args.hm_whole_volume_skeleton)
        if use_whole_volume_hm_path:
            print(
                "Using hm whole-volume skeleton path: overlap consensus -> whole heatmap -> skeletonize -> upsample."
            )
        metric_downsample_to_grid = bool(args.downsample_gt and int(args.output_patch_size) > 1)
        metric_coord_scale = float(args.output_patch_size) if metric_downsample_to_grid else 1.0
        if metric_downsample_to_grid:
            print(
                f"Using downsampled voxel-grid evaluation space with scale={metric_coord_scale:g}."
            )
        distance_metrics = {
            _radius_key(radius): {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
            for radius in eval_radii
        }
        path_distance_keys = [_radius_key(d) for d in path_connectivity_distances]
        r_tolerant_metric_counts = {
            _radius_key(radius): _new_r_tolerant_metric_counts(path_distance_keys)
            for radius in eval_radii
        }
        radius_metric_items = [
            (_radius_key(radius), float(radius) / metric_coord_scale) for radius in eval_radii
        ]
        path_length_metric_items = [
            (_radius_key(distance), float(distance) / metric_coord_scale)
            for distance in path_connectivity_distances
        ]
        match_metrics = {
            "one_to_one_pr": {"tp": 0, "fp": 0, "fn": 0},
            "merge_allowed": {"tp": 0, "fp": 0, "fn": 0},
            "split_allowed": {"tp": 0, "fp": 0, "fn": 0},
        }
        all_pred_dists = []
        all_gt_dists = []
        all_pred_inst_labels = []
        all_gt_inst_labels = []
        all_pred_neur_labels = []
        all_gt_neur_labels = []
        gt_skeleton_component_lengths = []
        gt_skeleton_component_point_counts = []
        gt_skeleton_length_stats_by_volume = []
        gt_skeleton_26cc_component_lengths = []
        gt_skeleton_26cc_component_point_counts = []
        gt_skeleton_26cc_length_stats_by_volume = []

        for vol_id, s in enumerate(data_vols):
            if args.fMOST:
                if "pretif" in s["subvol_folder"]:
                    voxel = pretif_open(
                        data_path / s["brain_id"] / s["axon_key"] / s["subvol_folder"]
                    )
                elif ".tif" in s["subvol_folder"]:
                    voxel = tifffile.imread(data_path / s["brain_id"] / s["subvol_folder"])
            else:
                voxel = np.array(
                    pretif_open(data_path / s["brain_id"] / s["subvol_folder"])[
                        s["range"][0][0] : s["range"][1][0],
                        s["range"][0][1] : s["range"][1][1],
                        s["range"][0][2] : s["range"][1][2],
                    ]
                )

            voxel, raw_neurites_dict, _, neuron_idx_mapping, _, cell_centers = convert_annotations(
                s, data_vol=voxel, fMOST=args.fMOST
            )
            viz_voxel = np.asarray(voxel, dtype=np.float32)
            voxel = np.expand_dims(voxel, axis=0).astype(np.float32)
            if args.twochannel:
                # concate a zero channel to make it 2-channel input for models trained with 2-channel input
                voxel = np.concatenate([np.zeros_like(voxel), voxel], axis=0)

            # center_crop, TODO: tiling for large volume
            transforms = ["whole_volume"]  # , "random_flip", "random_rotate_90"]
            if "norm" in model_name and "SNR" not in args.testdata:
                params = {"norm_values": (50, 200)}
                transforms.append("normalize_by_value")
            else:
                params = {"norm_percentiles": (float(args.p_min), float(args.p_max))}
            augmented = Transforms(voxel, raw_neurites_dict, transforms)  # , size=(32, 256, 256))
            augmented.get_params(params)
            augmented.apply_transform()
            augmented_neurites = augmented.coords
            voxel = augmented.volume

            target = {
                "annotations": {
                    "gt_neurites": [augmented_neurites,],
                    "gt_neurons": [neuron_idx_mapping,],
                    "cell_centers": [cell_centers,],
                }
            }
            volume_skeleton_length_stats = _skeleton_component_length_stats_from_target(target)
            volume_skeleton_length_stats["volume_id"] = int(vol_id)
            for meta_key in ("brain_id", "axon_key", "subvol_folder"):
                if meta_key in s:
                    volume_skeleton_length_stats[meta_key] = str(s.get(meta_key))
            gt_skeleton_length_stats_by_volume.append(volume_skeleton_length_stats)
            gt_skeleton_component_lengths.extend(volume_skeleton_length_stats["component_lengths"])
            gt_skeleton_component_point_counts.extend(
                volume_skeleton_length_stats.get("point_counts", [])
            )
            volume_skeleton_26cc_length_stats = _skeleton_26cc_length_stats_from_target(target)
            volume_skeleton_26cc_length_stats["volume_id"] = int(vol_id)
            for meta_key in ("brain_id", "axon_key", "subvol_folder"):
                if meta_key in s:
                    volume_skeleton_26cc_length_stats[meta_key] = str(s.get(meta_key))
            gt_skeleton_26cc_length_stats_by_volume.append(volume_skeleton_26cc_length_stats)
            gt_skeleton_26cc_component_lengths.extend(
                volume_skeleton_26cc_length_stats["component_lengths"]
            )
            gt_skeleton_26cc_component_point_counts.extend(
                volume_skeleton_26cc_length_stats.get("point_counts", [])
            )

            def _to_cpu(obj):
                if torch.is_tensor(obj):
                    return obj.cpu()
                if isinstance(obj, list):
                    return [_to_cpu(o) for o in obj]
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                return obj

            def _group_points_by_instance(points_t, inst_ids_t, mask_t):
                if mask_t is None:
                    mask_t = torch.ones(points_t.shape[0], dtype=torch.bool)
                if mask_t.sum().item() == 0:
                    return {}
                pts = points_t[mask_t].detach().cpu().numpy()
                inst_ids = inst_ids_t[mask_t].detach().cpu().numpy()
                grouped = defaultdict(list)
                for pt, inst_id in zip(pts, inst_ids):
                    grouped[int(inst_id)].append(pt)
                return {inst_id: np.asarray(pts, dtype=float) for inst_id, pts in grouped.items()}

            def _match_overlap_instances(prev_pts_by_id, curr_pts_by_id, dist_thr, ratio_thr):
                matches = {}
                if not prev_pts_by_id or not curr_pts_by_id:
                    return matches
                dist_thr = float(dist_thr)
                ratio_thr = float(ratio_thr)
                for curr_id, curr_pts in curr_pts_by_id.items():
                    if curr_pts.size == 0:
                        continue
                    best = None
                    best_score = -1.0
                    tree_curr = cKDTree(curr_pts)
                    for prev_id, prev_pts in prev_pts_by_id.items():
                        if prev_pts.size == 0:
                            continue
                        tree_prev = cKDTree(prev_pts)
                        d_curr, _ = tree_prev.query(curr_pts, k=1)
                        d_prev, _ = tree_curr.query(prev_pts, k=1)
                        match_curr = float(np.sum(d_curr <= dist_thr))
                        match_prev = float(np.sum(d_prev <= dist_thr))
                        curr_count = float(curr_pts.shape[0])
                        prev_count = float(prev_pts.shape[0])
                        if curr_count <= prev_count:
                            ratio_small = match_curr / max(1.0, curr_count)
                        else:
                            ratio_small = match_prev / max(1.0, prev_count)
                        if ratio_small >= ratio_thr and ratio_small > best_score:
                            best_score = ratio_small
                            best = prev_id
                    if best is not None:
                        matches[curr_id] = (best, best_score)
                return matches

            def _resolve_instance_merges(candidate_map):
                resolved = {}
                for curr_id, candidates in candidate_map.items():
                    if not candidates:
                        continue
                    best_score = -1.0
                    best_prev = None
                    for score, prev_id in candidates:
                        if score > best_score:
                            best_score = score
                            best_prev = prev_id
                    if best_prev is not None:
                        resolved[curr_id] = best_prev
                return resolved

            def _canonicalize_id_map(points_by_id, find_root):
                if not points_by_id:
                    return {}
                merged = defaultdict(list)
                for inst_id, pts in points_by_id.items():
                    root = find_root(inst_id)
                    merged[root].append(pts)
                out = {}
                for inst_id, pts_list in merged.items():
                    if len(pts_list) == 1:
                        out[inst_id] = pts_list[0]
                    else:
                        out[inst_id] = np.concatenate(pts_list, axis=0)
                return out

            [z, y, x] = voxel.shape[-3:]
            whole_boxes = []
            whole_labels = []
            whole_instance_ids = []
            whole_neuron_ids = []
            whole_logits = []
            whole_pred_hm = np.zeros(
                (
                    z // args.output_patch_size,
                    y // args.output_patch_size,
                    x // args.output_patch_size,
                ),
                dtype=np.float32,
            )
            whole_hm_positive_votes = (
                np.zeros_like(whole_pred_hm, dtype=np.uint16) if use_whole_volume_hm_path else None
            )
            whole_hm_cover_votes = (
                np.zeros_like(whole_pred_hm, dtype=np.uint16) if use_whole_volume_hm_path else None
            )
            whole_hm_consensus_mask = None
            instance_offset = 0

            overlap = args.overlap
            if any(o > 0 for o in overlap):
                stride_z = args.input_size[0] - overlap[0]
                stride_y = args.input_size[1] - overlap[1]
                stride_x = args.input_size[2] - overlap[2]
                if stride_z <= 0 or stride_y <= 0 or stride_x <= 0:
                    raise ValueError(
                        f"overlap={overlap} must be smaller than input_size={args.input_size}"
                    )
                z_tiles = (z + stride_z - 1) // stride_z
                y_tiles = (y + stride_y - 1) // stride_y
                x_tiles = (x + stride_x - 1) // stride_x
                left_z = overlap[0] // 2
                left_y = overlap[1] // 2
                left_x = overlap[2] // 2
                tile_overlap_stash = {}
                uf_parent = {}

                def _uf_find(x):
                    x = int(x)
                    if x not in uf_parent:
                        uf_parent[x] = x
                        return x
                    while uf_parent[x] != x:
                        uf_parent[x] = uf_parent[uf_parent[x]]
                        x = uf_parent[x]
                    return x

                def _uf_union(a, b):
                    ra = _uf_find(a)
                    rb = _uf_find(b)
                    if ra != rb:
                        uf_parent[rb] = ra

            else:
                z_tiles = (z - 1) // args.input_size[0] + 1
                y_tiles = (y - 1) // args.input_size[1] + 1
                x_tiles = (x - 1) // args.input_size[2] + 1
            for i in range(z_tiles):
                for j in range(y_tiles):
                    for k in range(x_tiles):
                        if any(o > 0 for o in overlap):
                            z0 = i * stride_z - left_z
                            y0 = j * stride_y - left_y
                            x0 = k * stride_x - left_x
                            z1 = z0 + args.input_size[0]
                            y1 = y0 + args.input_size[1]
                            x1 = x0 + args.input_size[2]

                            z_in_start = max(z0, 0)
                            y_in_start = max(y0, 0)
                            x_in_start = max(x0, 0)
                            z_in_end = min(z1, z)
                            y_in_end = min(y1, y)
                            x_in_end = min(x1, x)

                            z_out_start = z_in_start - z0
                            y_out_start = y_in_start - y0
                            x_out_start = x_in_start - x0
                            z_out_end = z_out_start + (z_in_end - z_in_start)
                            y_out_end = y_out_start + (y_in_end - y_in_start)
                            x_out_end = x_out_start + (x_in_end - x_in_start)
                            input_voxel = _extract_patch_with_tail_pad(
                                augmented.volume, (z0, y0, x0), args.input_size,
                            )
                        else:
                            z_start, z_end = crop_range(i, args.input_size[0], z)
                            y_start, y_end = crop_range(j, args.input_size[1], y)
                            x_start, x_end = crop_range(k, args.input_size[2], x)
                            input_voxel = _extract_patch_with_tail_pad(
                                augmented.volume, (z_start, y_start, x_start), args.input_size,
                            )
                        input_voxel = torch.tensor(input_voxel, dtype=torch.float32).unsqueeze(
                            0
                        )  # add batch dim
                        input_voxel = input_voxel.to(device, non_blocking=True)
                        model.eval()
                        dataset_idx = torch.zeros(1).to(device)

                        if model_arch in ["mae_vit_base_patch16_4"]:
                            # pad a zero to first channel
                            input_voxel = torch.cat(
                                [torch.zeros_like(input_voxel[:, :1]), input_voxel], dim=1
                            )

                        with torch.no_grad():
                            outputs = model(input_voxel, dataset_idx)
                        outputs = _to_cpu(outputs)
                        if "pred_hm" in outputs:
                            hm_arr = outputs["pred_hm"]
                            if torch.is_tensor(hm_arr):
                                hm_arr = hm_arr.numpy()
                            hm_arr = np.asarray(hm_arr)
                            if hm_arr.ndim == 5:
                                hm_arr = hm_arr[0, 0]
                            elif hm_arr.ndim == 4 and hm_arr.shape[0] == 1:
                                hm_arr = hm_arr[0]
                            dest_pos = None
                            dest_cov = None
                            if any(o > 0 for o in overlap):
                                z_hm_in_start = z_in_start // args.output_patch_size
                                y_hm_in_start = y_in_start // args.output_patch_size
                                x_hm_in_start = x_in_start // args.output_patch_size
                                z_hm_in_end = z_in_end // args.output_patch_size
                                y_hm_in_end = y_in_end // args.output_patch_size
                                x_hm_in_end = x_in_end // args.output_patch_size
                                z_hm_out_start = z_out_start // args.output_patch_size
                                y_hm_out_start = y_out_start // args.output_patch_size
                                x_hm_out_start = x_out_start // args.output_patch_size
                                z_hm_out_end = z_out_end // args.output_patch_size
                                y_hm_out_end = y_out_end // args.output_patch_size
                                x_hm_out_end = x_out_end // args.output_patch_size
                                hm_crop = hm_arr[
                                    z_hm_out_start:z_hm_out_end,
                                    y_hm_out_start:y_hm_out_end,
                                    x_hm_out_start:x_hm_out_end,
                                ]
                                dest = whole_pred_hm[
                                    z_hm_in_start:z_hm_in_end,
                                    y_hm_in_start:y_hm_in_end,
                                    x_hm_in_start:x_hm_in_end,
                                ]
                                if use_whole_volume_hm_path:
                                    dest_pos = whole_hm_positive_votes[
                                        z_hm_in_start:z_hm_in_end,
                                        y_hm_in_start:y_hm_in_end,
                                        x_hm_in_start:x_hm_in_end,
                                    ]
                                    dest_cov = whole_hm_cover_votes[
                                        z_hm_in_start:z_hm_in_end,
                                        y_hm_in_start:y_hm_in_end,
                                        x_hm_in_start:x_hm_in_end,
                                    ]
                            else:
                                z_sz = z_end - z_start
                                y_sz = y_end - y_start
                                x_sz = x_end - x_start
                                hm_crop = hm_arr[
                                    : z_sz // args.output_patch_size,
                                    : y_sz // args.output_patch_size,
                                    : x_sz // args.output_patch_size,
                                ]
                                dest = whole_pred_hm[
                                    z_start
                                    // args.output_patch_size : z_end
                                    // args.output_patch_size,
                                    y_start
                                    // args.output_patch_size : y_end
                                    // args.output_patch_size,
                                    x_start
                                    // args.output_patch_size : x_end
                                    // args.output_patch_size,
                                ]
                                if use_whole_volume_hm_path:
                                    dest_pos = whole_hm_positive_votes[
                                        z_start
                                        // args.output_patch_size : z_end
                                        // args.output_patch_size,
                                        y_start
                                        // args.output_patch_size : y_end
                                        // args.output_patch_size,
                                        x_start
                                        // args.output_patch_size : x_end
                                        // args.output_patch_size,
                                    ]
                                    dest_cov = whole_hm_cover_votes[
                                        z_start
                                        // args.output_patch_size : z_end
                                        // args.output_patch_size,
                                        y_start
                                        // args.output_patch_size : y_end
                                        // args.output_patch_size,
                                        x_start
                                        // args.output_patch_size : x_end
                                        // args.output_patch_size,
                                    ]
                            if hm_crop is not None and hm_crop.size > 0:
                                np.maximum(dest, hm_crop, out=dest)
                                if dest_pos is not None and dest_cov is not None:
                                    dest_cov += 1
                                    dest_pos += (hm_crop > args.hm_threshold).astype(
                                        dest_pos.dtype, copy=False
                                    )
                        if use_whole_volume_hm_path:
                            continue
                        (
                            outputs["pred_boxes"],
                            outputs["pred_logits"],
                            outputs["pred_instances"],
                        ) = pred_to_boxes(outputs, args.hm_threshold)
                        pred_hm = outputs["pred_hm"]
                        pred_inst = outputs["pred_inst"]
                        if pred_hm.dim() == 5 and pred_hm.size(1) == 1:
                            pred_hm = pred_hm[:, 0]
                        hm_mask = pred_hm > args.hm_threshold
                        if hm_mask.sum() == 0:
                            boxes = outputs["pred_boxes"][:0]
                            logits = outputs["pred_logits"][:0]
                            labels = torch.zeros((0,), dtype=torch.long)
                            instance_ids = torch.zeros((0,), dtype=torch.long)
                            neuron_ids = torch.zeros((0,), dtype=torch.long)
                        else:
                            instance_map = assign_instance_ids(
                                outputs["pred_hm"],
                                pred_inst,
                                hm_threshold=args.hm_threshold,
                                radius=args.instance_id_R,
                            )
                            boxes = outputs["pred_boxes"]
                            logits = outputs["pred_logits"]
                            labels = (
                                torch.argmax(logits, dim=1)
                                if logits.numel() > 0
                                else torch.zeros((0,), dtype=torch.long)
                            )
                            instance_ids = instance_map[hm_mask].view(-1)
                            neuron_ids = torch.full_like(instance_ids, -1, dtype=torch.long)
                        num_boxes_raw = labels.shape[0]
                        if num_boxes_raw > 0:
                            if any(o > 0 for o in overlap):
                                valid_start = [z_out_start, y_out_start, x_out_start]
                                valid_end = [z_out_end, y_out_end, x_out_end]
                            else:
                                valid_start = [0, 0, 0]
                                valid_end = [z_end - z_start, y_end - y_start, x_end - x_start]
                            (
                                boxes,
                                labels,
                                logits,
                                instance_ids,
                                neuron_ids,
                            ) = _filter_boxes_by_local_bounds(
                                boxes,
                                labels,
                                logits,
                                instance_ids,
                                neuron_ids,
                                args.input_size,
                                valid_start,
                                valid_end,
                            )
                        num_boxes = labels.shape[0]
                        if any(o > 0 for o in overlap) and num_boxes > 0:
                            (
                                boxes,
                                labels,
                                logits,
                                instance_ids,
                                removed_by_vote,
                            ) = overlap_filter_and_merge(
                                boxes_tile_norm=boxes,
                                labels=labels,
                                logits=logits,
                                inst_ids_tile=instance_ids,
                                tile_origin_zyx=[z0, y0, x0],
                                input_size_zyx=args.input_size,
                                vol_size_zyx=[z, y, x],
                                overlap_zyx=overlap,
                                instance_offset=instance_offset,
                                uf_find=_uf_find,
                                uf_union=_uf_union,
                                tile_overlap_stash=tile_overlap_stash,
                                tile_key=(i, j, k),
                                merge_dist=args.merge_dist,
                                vote_k=args.overlap_vote_k,
                                ratio_thr=args.overlap_vote_ratio,
                                require_two_sides=args.overlap_vote_two_sides,
                            )
                        elif all(o == 0 for o in overlap):
                            # plus the offset i j k to torch
                            boxes += torch.tensor([i, j, k, 0, 0, 0])
                        if all(o == 0 for o in overlap):
                            instance_ids += torch.tensor(instance_offset)

                        if boxes.shape[0] > 0:
                            whole_boxes.extend(boxes)
                            whole_labels.extend(labels)
                            whole_instance_ids.extend(instance_ids)
                            whole_neuron_ids.extend(neuron_ids)
                            whole_logits.extend(logits)

                        instance_offset += num_boxes_raw
                        if all(o == 0 for o in overlap):
                            x_ceil = k + 1
                            y_ceil = j + 1
                            z_ceil = i + 1
            if all(o == 0 for o in overlap):
                z_ceil = z_tiles
                y_ceil = y_tiles
                x_ceil = x_tiles
            if use_whole_volume_hm_path:
                raw_pred, whole_hm_consensus_mask = _build_whole_hm_centerline_pred(
                    whole_pred_hm,
                    whole_hm_positive_votes,
                    whole_hm_cover_votes,
                    args.hm_threshold,
                    full_shape=(z, y, x),
                    output_patch_size=args.output_patch_size,
                )
                cleaned_pred = None

                print(
                    f"Predicted {raw_pred['centers'].shape[0]} centerline voxels from whole-volume hm skeleton path."
                )
            else:
                print(f"Predicted {len(whole_boxes)} boxes/points in the whole volume.")
                boxes = (
                    torch.stack(whole_boxes, dim=0)
                    if len(whole_boxes) > 0
                    else torch.zeros((0, 6), dtype=torch.float32)
                )
                labels = torch.tensor(whole_labels, dtype=torch.int64)
                instance_ids = torch.tensor(whole_instance_ids, dtype=torch.int64)
                neuron_ids = torch.tensor(whole_neuron_ids, dtype=torch.int64)
                if len(whole_logits) > 0:
                    if torch.is_tensor(whole_logits[0]):
                        logits = torch.stack(whole_logits, dim=0)
                    else:
                        logits = torch.tensor(whole_logits, dtype=torch.float32)
                else:
                    logits = torch.zeros((0, 2), dtype=torch.float32)

                if any(o > 0 for o in overlap) and instance_ids.numel() > 0:
                    ids_np = instance_ids.detach().cpu().numpy()
                    ids_np = np.asarray(
                        [_uf_find(int(inst_id)) for inst_id in ids_np], dtype=np.int64
                    )
                    instance_ids = torch.tensor(ids_np, dtype=torch.int64)

                if all(o == 0 for o in overlap):
                    # rescale boxes to 0-1
                    boxes /= torch.tensor(
                        [z_ceil, y_ceil, x_ceil, z_ceil, y_ceil, x_ceil], dtype=torch.float32
                    )
            if use_whole_volume_hm_path:
                voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0)  # add channel dim
            else:
                pred = {
                    "boxes": boxes,
                    "labels": labels,
                    "instances": instance_ids,
                    "neurons": neuron_ids,
                    "logits": logits,
                }

                voxel = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0)  # add channel dim
                if any((o > 0 for o in overlap)):
                    whole_input_size = (z, y, x)
                else:
                    whole_input_size = (
                        z_ceil * args.input_size[0],
                        y_ceil * args.input_size[1],
                        x_ceil * args.input_size[2],
                    )

                normalized_pred = {}
                (
                    normalized_pred["centers"],
                    normalized_pred["inst_ids"],
                    normalized_pred["neuron_ids"],
                ) = _normalize_pred_inputs(pred, whole_input_size)

                trace_instance = False
                merge_all_instance = True
                cleaned_pred, raw_pred = skeletonize_pred(
                    normalized_pred,
                    whole_input_size,
                    output_patch_size=args.output_patch_size,
                    max_dist=args.merge_dist,
                    min_overlap_length=args.min_overlap_length,
                    merge=args.merge_instance,
                    crossing_min_component_length=args.min_seg_length,
                    break_branch=args.break_branch,
                    return_debug_info=False,
                    R_bridge=2 * args.merge_dist,
                    post_break_min_length=args.min_seg_length,
                    merge_overlap=True,
                    merge_tip=True,
                    trace_instance=trace_instance,
                    merge_all_instance=merge_all_instance,
                )

            pred_for_instance_eval = cleaned_pred if isinstance(cleaned_pred, dict) else raw_pred
            if not isinstance(pred_for_instance_eval, dict):
                pred_for_instance_eval = {
                    "centers": np.zeros((0, 3), dtype=float),
                    "inst_ids": np.zeros((0,), dtype=int),
                    "neuron_ids": np.zeros((0,), dtype=int),
                }

            primary_eval_radius_metrics = float(primary_eval_radius) / metric_coord_scale
            exclude_nuclei_metrics = (
                float(args.exclude_nuclei) / metric_coord_scale if args.exclude_nuclei > 0 else 0.0
            )

            metrics_esa = esa(
                raw_pred,
                target,
                distance_threshold=primary_eval_radius_metrics,
                exclude_nuclei=exclude_nuclei_metrics,
                coord_scale=metric_coord_scale,
                downsample_to_grid=metric_downsample_to_grid,
            )
            metrics_esa_viz = metrics_esa
            if metric_downsample_to_grid:
                metrics_esa_viz = esa(
                    raw_pred,
                    target,
                    distance_threshold=primary_eval_radius,
                    exclude_nuclei=args.exclude_nuclei,
                    coord_scale=1.0,
                    downsample_to_grid=False,
                )

            gt_points, gt_inst, _ = extract_gt_points(
                target,
                coord_scale=metric_coord_scale,
                downsample_to_grid=metric_downsample_to_grid,
            )
            pred_centers = _as_point_coords(pred_for_instance_eval.get("centers"), dtype=float)
            pred_inst = _as_numpy(pred_for_instance_eval.get("inst_ids"), dtype=int)
            pred_neur = _as_numpy(pred_for_instance_eval.get("neuron_ids"), dtype=int)
            if pred_centers is None:
                pred_centers = np.zeros((0, 3), dtype=float)
            if pred_inst is None or pred_inst.size != pred_centers.shape[0]:
                pred_inst = np.full((pred_centers.shape[0],), -1, dtype=int)
            if pred_neur is None or pred_neur.size != pred_centers.shape[0]:
                pred_neur = np.full((pred_centers.shape[0],), -1, dtype=int)
            if metric_downsample_to_grid and pred_centers.size > 0:
                pred_centers = _downsample_points_to_grid(
                    pred_centers, coord_scale=metric_coord_scale
                )
                pred_centers, pred_inst, pred_neur = _dedupe_labeled_points(
                    pred_centers, pred_inst, pred_neur
                )
            pred_dict = _group_by_instance(pred_centers, pred_inst)
            gt_dict = _group_by_instance(gt_points, gt_inst)
            metrics_match = evaluate_instance_matching_metrics_merge_and_split(
                pred_dict,
                gt_dict,
                r=primary_eval_radius_metrics,
                cost_thr=(float(args.match_cost_thr) / metric_coord_scale)
                if args.match_cost_thr is not None
                else None,
                tau_p=args.match_tau_p,
                tau_g=args.match_tau_g,
            )
            for key in ("one_to_one_pr", "merge_allowed", "split_allowed"):
                info = metrics_match.get(key, {})
                match_metrics[key]["tp"] += int(info.get("tp", 0))
                match_metrics[key]["fp"] += int(info.get("fp", 0))
                match_metrics[key]["fn"] += int(info.get("fn", 0))

            volume_r_tolerant_metrics = r_tolerant_erl_and_path_connectivity(
                raw_pred,
                target,
                radius_items=radius_metric_items,
                path_length_items=path_length_metric_items,
                coord_scale=metric_coord_scale,
                downsample_to_grid=metric_downsample_to_grid,
            )
            _accumulate_r_tolerant_metric_counts(
                r_tolerant_metric_counts, volume_r_tolerant_metrics
            )

            for radius in eval_radii:
                r_key = _radius_key(radius)
                radius_metrics = float(radius) / metric_coord_scale
                metrics_for_radius = metrics_esa
                if args.exclude_nuclei > 0 and radius_metrics != primary_eval_radius_metrics:
                    metrics_for_radius = esa(
                        raw_pred,
                        target,
                        distance_threshold=radius_metrics,
                        exclude_nuclei=exclude_nuclei_metrics,
                        coord_scale=metric_coord_scale,
                        downsample_to_grid=metric_downsample_to_grid,
                    )
                pred_dists = np.asarray(metrics_for_radius["pred_to_gt_dists"], dtype=float)
                gt_dists = np.asarray(metrics_for_radius["gt_to_pred_dists"], dtype=float)
                pred_within = int(np.sum(pred_dists <= radius_metrics)) if pred_dists.size else 0
                gt_within = int(np.sum(gt_dists <= radius_metrics)) if gt_dists.size else 0
                distance_metrics[r_key]["tp"] += pred_within
                distance_metrics[r_key]["fp"] += metrics_for_radius["num_pred"] - pred_within
                distance_metrics[r_key]["fn"] += metrics_for_radius["num_gt"] - gt_within
                distance_metrics[r_key]["gt"] += metrics_for_radius["num_gt"]
                distance_metrics[r_key]["pred"] += metrics_for_radius["num_pred"]
            pred_dists = np.asarray(metrics_esa["pred_to_gt_dists"], dtype=float)
            gt_dists = np.asarray(metrics_esa["gt_to_pred_dists"], dtype=float)
            all_pred_dists.extend((pred_dists * metric_coord_scale).tolist())
            all_gt_dists.extend((gt_dists * metric_coord_scale).tolist())
            all_pred_inst_labels.extend(metrics_esa["pred_labels"]["instance"])
            all_pred_neur_labels.extend(metrics_esa["pred_labels"]["neuron"])
            all_gt_inst_labels.extend(metrics_esa["gt_labels_for_preds"]["instance"])
            all_gt_neur_labels.extend(metrics_esa["gt_labels_for_preds"]["neuron"])

            # visualizations
            viz_dir = os.path.join(summary_viz_dir, f"data_{vol_id}")

            esa_viz_path = os.path.join(viz_dir, "esa_distances.png")

            # save data, target and pred
            save_data(viz_dir, voxel, target, pred_for_instance_eval)

            if whole_pred_hm is not None:
                hm_dir = Path(viz_dir)
                hm_dir.mkdir(parents=True, exist_ok=True)
                hm = np.asarray(whole_pred_hm, dtype=np.float32)
                if use_whole_volume_hm_path and whole_hm_consensus_mask is not None:
                    hm_mask = np.asarray(whole_hm_consensus_mask, dtype=np.uint8)
                else:
                    hm_mask = (hm > args.hm_threshold).astype(np.uint8)
                hm_clip = np.clip(hm, 0.0, 1.0)
                hm_u8 = (hm_clip * 255.0).round().astype(np.uint8)
                hm_u16 = (hm_clip * 65535.0).round().astype(np.uint16)
                hm_mask_u8 = (hm_mask * 255).astype(np.uint8)
                hm_mask_u16 = (hm_mask * 65535).astype(np.uint16)
                tifffile.imwrite(hm_dir / "pred_hm_u8.tif", hm_u8)
                tifffile.imwrite(hm_dir / "pred_hm_u16.tif", hm_u16)
                tifffile.imwrite(hm_dir / "pred_hm_mask_u8.tif", hm_mask_u8)
                tifffile.imwrite(hm_dir / "pred_hm_mask_u16.tif", hm_mask_u16)

                ps = int(args.output_patch_size)
                if ps > 1:
                    hm_up = np.repeat(np.repeat(np.repeat(hm, ps, axis=0), ps, axis=1), ps, axis=2)
                    hm_mask_up = np.repeat(
                        np.repeat(np.repeat(hm_mask, ps, axis=0), ps, axis=1), ps, axis=2
                    )
                    hm_up = hm_up[:z, :y, :x]
                    hm_mask_up = hm_mask_up[:z, :y, :x]
                else:
                    hm_up = hm
                    hm_mask_up = hm_mask
                hm_up_clip = np.clip(hm_up, 0.0, 1.0)
                hm_up_u8 = (hm_up_clip * 255.0).round().astype(np.uint8)
                hm_up_u16 = (hm_up_clip * 65535.0).round().astype(np.uint16)
                hm_mask_up_u8 = (hm_mask_up * 255).astype(np.uint8)
                hm_mask_up_u16 = (hm_mask_up * 65535).astype(np.uint16)
                tifffile.imwrite(hm_dir / "pred_hm_up_u8.tif", hm_up_u8)
                tifffile.imwrite(hm_dir / "pred_hm_up_u16.tif", hm_up_u16)
                tifffile.imwrite(hm_dir / "pred_hm_mask_up_u8.tif", hm_mask_up_u8)
                tifffile.imwrite(hm_dir / "pred_hm_mask_up_u16.tif", hm_mask_up_u16)

            viz_esa_distances(
                esa_viz_path,
                viz_voxel,
                target,
                raw_pred,
                metrics_esa_viz,
                pred_hm=whole_pred_hm,
                output_patch_size=args.output_patch_size,
                z_step=args.viz_z_step,
                dilation=args.thick_viz,
                exclude_nuclei=args.exclude_nuclei,
            )

        eps = 1e-8
        by_dist = {
            "tp": {},
            "fp": {},
            "fn": {},
            "precision": {},
            "recall": {},
            "f1": {},
        }
        for radius in eval_radii:
            r_key = _radius_key(radius)
            metrics_r = distance_metrics[r_key]
            precision = metrics_r["tp"] / (metrics_r["tp"] + metrics_r["fp"] + eps)
            recall = metrics_r["tp"] / (metrics_r["tp"] + metrics_r["fn"] + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            by_dist["tp"][r_key] = int(metrics_r["tp"])
            by_dist["fp"][r_key] = int(metrics_r["fp"])
            by_dist["fn"][r_key] = int(metrics_r["fn"])
            by_dist["precision"][r_key] = float(precision)
            by_dist["recall"][r_key] = float(recall)
            by_dist["f1"][r_key] = float(f1)

        r_tolerant_by_dist = _finalize_r_tolerant_metric_counts(
            r_tolerant_metric_counts, path_connectivity_distances, length_scale=metric_coord_scale,
        )
        gt_skeleton_length_stats = _summarize_skeleton_component_lengths(
            gt_skeleton_component_lengths,
            gt_skeleton_component_point_counts,
            length_source="aggregate_densified_gt_instances",
        )
        gt_skeleton_length_stats["per_volume"] = gt_skeleton_length_stats_by_volume
        gt_skeleton_26cc_length_stats = _summarize_skeleton_component_lengths(
            gt_skeleton_26cc_component_lengths,
            gt_skeleton_26cc_component_point_counts,
            length_source="aggregate_connected_components_26",
        )
        gt_skeleton_26cc_length_stats["per_volume"] = gt_skeleton_26cc_length_stats_by_volume

        finite_pred_dists = [d for d in all_pred_dists if np.isfinite(d)]
        finite_gt_dists = [d for d in all_gt_dists if np.isfinite(d)]
        mean_pred_dist = float(np.mean(finite_pred_dists)) if finite_pred_dists else float("nan")
        mean_gt_dist = float(np.mean(finite_gt_dists)) if finite_gt_dists else float("nan")
        mean_overall_dist = (
            0.5 * (mean_pred_dist + mean_gt_dist)
            if np.isfinite(mean_pred_dist) and np.isfinite(mean_gt_dist)
            else float("nan")
        )

        finite_ssd_pred_dists = [d for d in finite_pred_dists if d > primary_eval_radius]
        finite_ssd_gt_dists = [d for d in finite_gt_dists if d > primary_eval_radius]
        ssd_pred_dist = (
            float(np.mean(finite_ssd_pred_dists)) if finite_ssd_pred_dists else float("nan")
        )
        ssd_gt_dist = float(np.mean(finite_ssd_gt_dists)) if finite_ssd_gt_dists else float("nan")
        ssd_overall_dist = (
            0.5 * (ssd_pred_dist + ssd_gt_dist)
            if np.isfinite(ssd_pred_dist) and np.isfinite(ssd_gt_dist)
            else float("nan")
        )

        for radius in eval_radii:
            r_key = _radius_key(radius)
            metrics_r = distance_metrics[r_key]
            print(
                f'[radius {r_key}] precision={by_dist["precision"][r_key]:.4f} '
                f'recall={by_dist["recall"][r_key]:.4f} f1={by_dist["f1"][r_key]:.4f} '
                f'tp={metrics_r["tp"]} fp={metrics_r["fp"]} fn={metrics_r["fn"]} '
                f'pred_boxes={metrics_r["pred"]} gt_points={metrics_r["gt"]}'
            )
            erl_norm = r_tolerant_by_dist["r_tolerant_erl"]["normalized"][r_key]
            erl_expected = r_tolerant_by_dist["r_tolerant_erl"]["expected_run_length"][r_key]
            path_scores = r_tolerant_by_dist["r_tolerant_path_connectivity"]["score"][r_key]
            path_score_text = ", ".join(
                f"{d_key}:{score:.4f}" if np.isfinite(score) else f"{d_key}:nan"
                for d_key, score in path_scores.items()
            )
            print(
                f"[radius {r_key}] nERL_R={erl_norm:.4f} "
                f"ERL_R={erl_expected:.3f} path_connectivity={{{path_score_text}}}"
            )
        print(
            f"mean pred->gt distance: {mean_pred_dist:.3f}, mean gt->pred distance: {mean_gt_dist:.3f}\n mean bidirectional distance: {mean_overall_dist:.3f}"
        )
        print(
            f"SSD (r={primary_eval_radius:g}) pred->gt: {ssd_pred_dist:.3f}, "
            f"gt->pred: {ssd_gt_dist:.3f}, mean: {ssd_overall_dist:.3f}"
        )
        print(
            f"GT instance skeleton length: mean={gt_skeleton_length_stats['mean_length']:.3f}, "
            f"median={gt_skeleton_length_stats['median_length']:.3f}, "
            f"min={gt_skeleton_length_stats['min_length']:.3f}, "
            f"max={gt_skeleton_length_stats['max_length']:.3f}, "
            f"n={gt_skeleton_length_stats['num_components']}"
        )
        print(
            f"GT 26-connected skeleton component length: mean={gt_skeleton_26cc_length_stats['mean_length']:.3f}, "
            f"median={gt_skeleton_26cc_length_stats['median_length']:.3f}, "
            f"min={gt_skeleton_26cc_length_stats['min_length']:.3f}, "
            f"max={gt_skeleton_26cc_length_stats['max_length']:.3f}, "
            f"n={gt_skeleton_26cc_length_stats['num_components']}"
        )
        if all_pred_inst_labels:
            ari_instance = adjusted_rand_score(all_gt_inst_labels, all_pred_inst_labels)
            ari_neuron = adjusted_rand_score(all_gt_neur_labels, all_pred_neur_labels)
            print(f"ARI (instance): {ari_instance:.4f}, ARI (neuron): {ari_neuron:.4f}")
        else:
            print("ARI (instance/neuron): not available (no predictions).")
        for key, label in (
            ("one_to_one_pr", "one-to-one"),
            ("merge_allowed", "merge-allowed"),
            ("split_allowed", "split-allowed"),
        ):
            tp = match_metrics[key]["tp"]
            fp = match_metrics[key]["fp"]
            fn = match_metrics[key]["fn"]
            p, r, f1m = _prf(tp, fp, fn)
            print(
                f"[match {label}] precision={p:.4f} recall={r:.4f} f1={f1m:.4f} "
                f"tp={tp} fp={fp} fn={fn}"
            )

        primary_radius_key = _radius_key(eval_radii[0])
        primary_metrics = distance_metrics[primary_radius_key]
        primary_erl = {
            key: values[primary_radius_key]
            for key, values in r_tolerant_by_dist["r_tolerant_erl"].items()
        }
        primary_path_connectivity = {
            "distances": r_tolerant_by_dist["r_tolerant_path_connectivity"]["distances"],
            "distance_unit": r_tolerant_by_dist["r_tolerant_path_connectivity"]["distance_unit"],
            "distance_definition": r_tolerant_by_dist["r_tolerant_path_connectivity"][
                "distance_definition"
            ],
            "score": r_tolerant_by_dist["r_tolerant_path_connectivity"]["score"][
                primary_radius_key
            ],
            "connected": r_tolerant_by_dist["r_tolerant_path_connectivity"]["connected"][
                primary_radius_key
            ],
            "total": r_tolerant_by_dist["r_tolerant_path_connectivity"]["total"][
                primary_radius_key
            ],
        }
        model_threshold_results[model_name] = {
            "hm_threshold": float(args.hm_threshold),
            "eval_radius": float(eval_radii[0]),
            "precision": float(by_dist["precision"][primary_radius_key]),
            "recall": float(by_dist["recall"][primary_radius_key]),
            "f1": float(by_dist["f1"][primary_radius_key]),
            "tp": int(primary_metrics["tp"]),
            "fp": int(primary_metrics["fp"]),
            "fn": int(primary_metrics["fn"]),
            "r_tolerant_erl": primary_erl,
            "r_tolerant_path_connectivity": primary_path_connectivity,
            "gt_skeleton_length_stats": gt_skeleton_length_stats,
            "gt_skeleton_26cc_length_stats": gt_skeleton_26cc_length_stats,
            "summary_viz_dir": summary_viz_dir,
            "hm_eval_dir": hm_eval_dir,
        }
        os.makedirs(summary_viz_dir, exist_ok=True)
        if len(eval_radii) > 1:
            by_dist_path = os.path.join(summary_viz_dir, "by_dist.json")
            by_dist_payload = {
                "eval_radius": [float(r) for r in eval_radii],
                "tp": by_dist["tp"],
                "fp": by_dist["fp"],
                "fn": by_dist["fn"],
                "precision": by_dist["precision"],
                "recall": by_dist["recall"],
                "f1": by_dist["f1"],
                "r_tolerant_erl": r_tolerant_by_dist["r_tolerant_erl"],
                "r_tolerant_path_connectivity": r_tolerant_by_dist["r_tolerant_path_connectivity"],
                "gt_skeleton_length_stats": gt_skeleton_length_stats,
                "gt_skeleton_26cc_length_stats": gt_skeleton_26cc_length_stats,
                "esa": {"p2g": mean_pred_dist, "g2p": mean_gt_dist, "mean": mean_overall_dist,},
                "ssd": {
                    "radius": float(primary_eval_radius),
                    "p2g": ssd_pred_dist,
                    "g2p": ssd_gt_dist,
                    "mean": ssd_overall_dist,
                },
            }
            with open(by_dist_path, "w") as f:
                json.dump(by_dist_payload, f, indent=2)
            print(f"Saved per-radius metrics to {by_dist_path}")
    return model_threshold_results


def main(args):
    hm_thresholds = _parse_hm_thresholds(args.hm_threshold)
    eval_radii = _parse_eval_radii(args.eval_radius)
    args.path_connectivity_distances = _parse_positive_distances(
        args.path_connectivity_distances, "path_connectivity_distances",
    )
    assert not (len(hm_thresholds) > 1 and len(eval_radii) > 1), (
        f"hm_threshold and eval_radius cannot both have length > 1; "
        f"got len(hm_threshold)={len(hm_thresholds)}, len(eval_radius)={len(eval_radii)}"
    )

    if args.check_dataset:
        data_path = Path(args.data_path)
        data_vols = load_annotations_from_json(
            args.testdata,
            data_path,
            labeled=args.labeled,
            selected_axon_id=args.selected_axon_id,
            fMOST=args.fMOST,
        )
        check_dataset(data_path, data_vols, args.selected_axon_id)
        return

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d-%H%M", current_time)
    hm_eval_by_model = defaultdict(dict)
    hm_eval_dir_by_model = {}

    for hm_threshold in hm_thresholds:
        results = _run_single_hm_threshold(args, hm_threshold, eval_radii, timestamp)
        hm_key = _radius_key(hm_threshold)
        for model_name, metrics in results.items():
            hm_eval_by_model[model_name][hm_key] = {
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "r_tolerant_erl": metrics["r_tolerant_erl"],
                "r_tolerant_path_connectivity": metrics["r_tolerant_path_connectivity"],
                "gt_skeleton_length_stats": metrics["gt_skeleton_length_stats"],
                "gt_skeleton_26cc_length_stats": metrics["gt_skeleton_26cc_length_stats"],
            }
            hm_eval_dir_by_model[model_name] = metrics["hm_eval_dir"]

    for model_name, per_hm_metrics in hm_eval_by_model.items():
        if not per_hm_metrics:
            continue
        best_hm_key, best_metrics = max(per_hm_metrics.items(), key=lambda kv: kv[1]["f1"])
        hm_eval_payload = {
            "eval_radius_for_hm_rank": float(eval_radii[0]),
            "best_hm_threshold": float(best_hm_key),
            "best_f1": float(best_metrics["f1"]),
            "gt_skeleton_length_stats": best_metrics["gt_skeleton_length_stats"],
            "gt_skeleton_26cc_length_stats": best_metrics["gt_skeleton_26cc_length_stats"],
            "metrics_by_hm_threshold": per_hm_metrics,
        }
        hm_eval_dir = hm_eval_dir_by_model.get(model_name)
        if hm_eval_dir is None:
            continue
        os.makedirs(hm_eval_dir, exist_ok=True)
        hm_eval_path = os.path.join(hm_eval_dir, "hm_eval.json")
        with open(hm_eval_path, "w") as f:
            json.dump(hm_eval_payload, f, indent=2)
        print(f"Saved hm-threshold sweep metrics to {hm_eval_path}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
