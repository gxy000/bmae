"""Load axon evaluation annotations and convert SWC/list coordinates to volume space."""

import random
import torch
import tifffile
from torch.utils.data import Dataset
from .bmae_utils import pretif_open, connect_neurite_points
from .transforms import Transforms

import json
from pathlib import Path

import numpy as np

from neurom import iter_sections
import warnings

warnings.filterwarnings("ignore")
import morphio
from morphio import Morphology as MorphioMorph, Option
from neurom.io.utils import load_morphology as nm_load_morphology

morphio.set_maximum_warnings(0)


def _safe_load_morphology(swc_path):
    """Robustly load an SWC while allowing section-type changes when supported."""
    option = None
    for opt_name in ("unifurcated_section_change", "allow_unifurcated_section_change"):
        option = getattr(Option, opt_name, None)
        if option is not None:
            break

    morphio_kwargs = {"options": option} if option is not None else {}
    m = MorphioMorph(swc_path, **morphio_kwargs)
    # give neurom a preconstructed morphio object
    return nm_load_morphology(m)


def fetch_points(
    neurite_idx,
    neuron_idx,
    neuron_idx_mapping,
    neuron,
    zyx_min=(0, 0, 0),
    subvol_size=(400, 1000, 1000),
    annotation_offset=(10, 0, 0),
    ntype=(3, 4),
):
    # fetch for the subvol, and then random crop
    offset = np.array(zyx_min, dtype=float) - np.array(annotation_offset)
    max_bound = np.array(subvol_size, dtype=float)
    neurites_dict = {}
    total_points = 0
    section_types = set()
    for section in iter_sections(neuron.neurites):
        # only care axon
        section_types.add(section.type)
        if section.type not in ntype:
            continue
        points = np.array(section.points[:, 2::-1]) - offset
        points = points[np.all((points >= 0) & (points < max_bound), axis=1)]
        if points.shape[0] < 2:
            continue
        neurites_dict[neurite_idx] = points
        neuron_idx_mapping[neurite_idx] = neuron_idx
        total_points += points.shape[0]
        neurite_idx += 1
    neuron_centers = []
    try:
        soma = neuron.soma.center[::-1] - offset
        if np.all((soma >= 0) & (soma < max_bound)):
            neuron_centers = [soma]
    except Exception:
        neuron_centers = []
    return neurites_dict, neuron_idx_mapping, neurite_idx, neuron_centers


def fetch_points_swctxt(
    neurite_idx,
    neuron_idx,
    neuron_idx_mapping,
    swc_data,
    zyx_min=(0, 0, 0),
    subvol_size=(400, 1000, 1000),
    annotation_offset=(10, 0, 0),
):
    offset = np.array(zyx_min, dtype=float) - np.array(annotation_offset)
    max_bound = np.array(subvol_size, dtype=float)
    neurites_dict = {}
    total_points = 0
    if swc_data.size == 0:
        return neurites_dict, neuron_idx, neuron_idx_mapping, neurite_idx, []

    node_ids = swc_data[:, 0].astype(int)
    parent_ids = swc_data[:, 6].astype(int)
    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

    children = {node_id: [] for node_id in node_ids}
    for node_id, parent_id in zip(node_ids, parent_ids):
        if parent_id == -1:
            continue
        if parent_id in children:
            children[parent_id].append(node_id)

    for node_id in children:
        children[node_id].sort()

    def point_of(node_id):
        idx = id_to_index[node_id]
        # swc columns: id, type, x, y, z, radius, parent; convert to z, y, x
        return swc_data[idx, 4:1:-1].astype(float) - offset

    def is_start(node_id):
        parent_id = parent_ids[id_to_index[node_id]]
        if parent_id == -1:
            return True
        parent_children = children.get(parent_id, [])
        return len(parent_children) > 1

    for node_id in node_ids:
        if not is_start(node_id):
            continue
        for child_id in children.get(node_id, []):
            points = [point_of(node_id), point_of(child_id)]
            cur = child_id
            while True:
                cur_children = children.get(cur, [])
                if len(cur_children) != 1:
                    break
                nxt = cur_children[0]
                points.append(point_of(nxt))
                cur = nxt
            points = np.array(points)
            points = points[np.all((points >= 0) & (points < max_bound), axis=1)]
            if points.shape[0] < 2:
                continue
            neurites_dict[neurite_idx] = points
            neuron_idx_mapping[neurite_idx] = neuron_idx
            total_points += points.shape[0]
            neurite_idx += 1

    return neurites_dict, neuron_idx, neuron_idx_mapping, neurite_idx, []


def fetch_points_from_list(
    neurite_idx,
    neuron_idx,
    neuron_idx_mapping,
    neurons,
    zyx_min=(0, 0, 0),
    subvol_size=(400, 1000, 1000),
    annotation_offset=(10, 0, 0),
    ntype=(3, 4),
):
    offset = np.array(zyx_min, dtype=float) - np.array(annotation_offset)
    max_bound = np.array(subvol_size, dtype=float)
    neurites_dict = {}
    total_points = 0
    cell_centers = []
    print(f"neurons keys: {neurons.keys()}")
    axons_raw = neurons["axons"]
    for center in neurons.get("centers", []):
        soma = np.asarray(center, dtype=float) - offset
        if np.all((soma >= 0) & (soma < max_bound)):
            cell_centers.append(soma)
    print(f"find {len(cell_centers)} cell centers.")
    for axon in axons_raw:
        soma = None
        for key, points in axon.items():
            points = np.array(points) - offset
            points = points[np.all((points >= 0) & (points < max_bound), axis=1)]
            if points.shape[0] < 2:
                continue
            neurites_dict[neurite_idx] = points
            neuron_idx_mapping[neurite_idx] = neuron_idx
            total_points += points.shape[0]
            neurite_idx += 1
        neuron_idx += 1
    return neurites_dict, neuron_idx, neuron_idx_mapping, neurite_idx, cell_centers


def load_annotations_from_json(
    json_path, data_path, labeled=True, selected_axon_id=None, fMOST=True
):
    data_path = Path(data_path)
    json_path = data_path / f"{json_path}.json"
    data_vols = []
    with open(json_path) as f:
        brains = json.load(f)
    if not isinstance(brains, list):
        brains = [brains]
    for brain in brains:
        brain_id = brain["brain_ID"]
        if not labeled:
            assert "range" in brain, f"brain {brain_id} does not have range"
            assert "marker_folder" in brain, f"brain {brain_id} does not have marker_folder"

            # TODO sliding window for range
            for vrange in brain["range"]:
                data_vols.append(
                    {
                        "brain_id": brain_id,
                        "marker_folder": brain["marker_folder"],
                        "range": vrange,
                    }
                )
        else:
            offset = brain["offset"]
            ntype = brain["ntype"] if "ntype" in brain else (3, 4)
            if fMOST:
                if selected_axon_id is not None:
                    for axon_id in selected_axon_id:
                        axon_key = f"axon_of_{axon_id}"
                        swc_rel = brain["annotation_dict"][axon_key]
                        swc_path = data_path / brain_id / "1" / swc_rel
                        for subvol_folder in brain["subvol_dict"][axon_key]:
                            data_vols.append(
                                {
                                    "brain_id": brain_id,
                                    "axon_key": axon_key,
                                    "swc_path": swc_path,
                                    "subvol_folder": subvol_folder,  # if "pretif" in subvol_folder else f"{subvol_folder}_pretif",
                                    "offset": offset,
                                    "ntype": ntype,
                                }
                            )
                else:
                    for axon_key, swc_rel in brain["annotation_dict"].items():
                        swc_path = data_path / brain_id / "1" / swc_rel
                        for subvol_folder in brain["subvol_dict"][axon_key]:
                            data_vols.append(
                                {
                                    "brain_id": brain_id,
                                    "axon_key": axon_key,
                                    "swc_path": swc_path,
                                    "subvol_folder": subvol_folder,  # if "pretif" in subvol_folder else f"{subvol_folder}_pretif",
                                    "offset": offset,
                                    "ntype": ntype,
                                }
                            )
            else:
                data_vols.append(
                    {
                        "brain_id": brain_id,
                        "subvol_folder": brain["marker_folder"],
                        "range": brain["range"],  # Note the boundary of the annotation #TODO
                        "axons": brain[
                            "axons"
                        ],  # list of dict of list, "starting points": List[np.ndarray]
                        "neurons": brain["neurons"],
                        "centers": brain.get("centers", []),
                        "offset": offset,
                        "ntype": (999),
                    }
                )
    return data_vols


def _extract_centers_from_info(
    info, zyx_min=(0, 0, 0), subvol_size=(400, 1000, 1000), annotation_offset=(10, 0, 0)
):
    offset = np.array(zyx_min, dtype=float) - np.array(annotation_offset)
    max_bound = np.array(subvol_size, dtype=float)
    cell_centers = []
    for center in info.get("centers", []):
        point = np.asarray(center, dtype=float)
        if point.size < 3:
            continue
        point = point[:3] - offset
        if np.all((point >= 0) & (point < max_bound)):
            cell_centers.append(point)
    return cell_centers


def convert_annotations(
    info,
    data_vol=None,
    _tif_cache=None,
    fMOST=False,
    neurite_idx=0,
    neuron_idx=0,
    neuron_idx_mapping={},
):
    assert (
        _tif_cache is not None or data_vol is not None
    ), "_tif_cache must be provided if data_vol is None"
    if fMOST:
        brain_id, axon_key, swc_path, subvol_folder, offset, ntype = (
            info["brain_id"],
            info["axon_key"],
            info["swc_path"],
            info["subvol_folder"],
            info["offset"][::-1],
            info["ntype"],
        )

        # load pretif volume (single‑channel); add channel dim
        if data_vol is None:
            data_vol = _tif_cache[(brain_id, axon_key, subvol_folder)]
        data_vol_size = data_vol.shape
        if "pretif" in subvol_folder:
            x_min, _, y_min, _, z_min, _ = [
                int(s) for s in subvol_folder.split("_") if s.isdigit()
            ]  # absolute xyz min of this sub‑vol
            neuron = _safe_load_morphology(swc_path)
            raw_neurites_dict, neuron_idx_mapping, neurite_idx, _unused_centers = fetch_points(
                neurite_idx,
                neuron_idx,
                neuron_idx_mapping,
                neuron,
                (z_min, y_min, x_min),
                data_vol_size,
                offset,
                ntype,
            )  # List[List[np.ndarray]]
        elif ".tif" in subvol_folder:
            x_min, y_min, z_min = 0, 0, 0
            swc_data = np.loadtxt(swc_path, ndmin=2)
            (
                raw_neurites_dict,
                neuron_idx,
                neuron_idx_mapping,
                neurite_idx,
                _unused_centers,
            ) = fetch_points_swctxt(
                neurite_idx,
                neuron_idx,
                neuron_idx_mapping,
                swc_data,
                (z_min, y_min, x_min),
                data_vol_size,
                offset,
            )
        cell_centers = _extract_centers_from_info(
            info, (z_min, y_min, x_min), data_vol_size, offset
        )
        neuron_idx += 1
    else:
        brain_id, subvol_folder, subvol_range, offset, ntype = (
            info["brain_id"],
            info["subvol_folder"],
            info["range"],
            info["offset"][::-1],
            info["ntype"],
        )

        # load pretif volume (single‑channel); add channel dim
        if data_vol is None:
            data_vol = _tif_cache[(brain_id, subvol_folder, "_".join(map(str, subvol_range)))]
        data_vol_size = data_vol.shape
        z_min, y_min, x_min = subvol_range[0]

        neurons = info
        (
            raw_neurites_dict,
            neuron_idx,
            neuron_idx_mapping,
            neurite_idx,
            _unused_centers,
        ) = fetch_points_from_list(
            neurite_idx,
            neuron_idx,
            neuron_idx_mapping,
            neurons,
            (z_min, y_min, x_min),
            data_vol_size,
            offset,
            ntype,
        )  # List[List[np.ndarray]]
        cell_centers = _extract_centers_from_info(
            info, (z_min, y_min, x_min), data_vol_size, offset
        )
    return data_vol, raw_neurites_dict, neuron_idx_mapping, neuron_idx, neurite_idx, cell_centers


def clip_segment_to_view(p1, p2, depth, height, width):
    """Clip a (z, y, x) segment to the current view using Liang-Barsky; returns None if fully outside."""
    if depth <= 0 or height <= 0 or width <= 0:
        return None

    z_min, z_max = 0.0, float(depth - 1)
    y_min, y_max = 0.0, float(height - 1)
    x_min, x_max = 0.0, float(width - 1)

    z1, y1, x1 = (float(c) for c in p1)
    z2, y2, x2 = (float(c) for c in p2)

    dz = z2 - z1
    dy = y2 - y1
    dx = x2 - x1

    p = (-dz, dz, -dy, dy, -dx, dx)
    q = (z1 - z_min, z_max - z1, y1 - y_min, y_max - y1, x1 - x_min, x_max - x1)

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0:
                return None
            continue
        r = qi / pi
        if pi < 0:
            if r > u2:
                return None
            if r > u1:
                u1 = r
        else:
            if r < u1:
                return None
            if r < u2:
                u2 = r

    start = (z1 + u1 * dz, y1 + u1 * dy, x1 + u1 * dx)
    end = (z1 + u2 * dz, y1 + u2 * dy, x1 + u2 * dx)
    return start, end


def gaussian3D(shape, sigma=1):
    l, m, n = [(ss - 1.0) / 2.0 for ss in shape]
    z, y, x = np.ogrid[-l : l + 1, -m : m + 1, -n : n + 1]
    h = np.exp(-(z * z + x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    # TODO sigma value
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter / 6)
    z, y, x = int(center[0]), int(center[1]), int(center[2])
    d, h, w = heatmap.shape[0:3]
    z_minus, z_plus = min(z, radius), min(d - z, radius + 1)
    y_minus, y_plus = min(y, radius), min(h - y, radius + 1)
    x_minus, x_plus = min(x, radius), min(w - x, radius + 1)
    masked_heatmap = heatmap[
        z - z_minus : z + z_plus, y - y_minus : y + y_plus, x - x_minus : x + x_plus
    ]
    masked_gaussian = gaussian[
        radius - z_minus : radius + z_plus,
        radius - y_minus : radius + y_plus,
        radius - x_minus : radius + x_plus,
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def center_to_gaussian(points, subvol_size, R, object_radius):
    """
    points: labeleed cell center
    subvol_size: the size of the input volume
    R: downsample factor in CenterNet
    """
    subvol_size = tuple(np.array(subvol_size) // R)
    heatmap = np.zeros(subvol_size)
    for point in points:
        heatmap = draw_umich_gaussian(heatmap, point // R, radius=object_radius // R)

    return heatmap


class AxonSegDataset(Dataset):
    """Dataset that serves 3‑D sub‑volumes and centernet‑style targets.

    Expected JSON entry format
    --------------------------
    {
        "brain_id" : "Chung_WT1_Syto16-NeuN-TH",
        "annotation_dict" : {"axon_of_0":xxx.swc, "axon_of_1":xxx.swc},
        "subvol_dict" : {
            "axon_of_0": ["subvol_1", "subvol_2", ...],
            "axon_of_1": ["subvol_1", "subvol_2", ...]
        }
    }
    where cz/cy/cx are absolute coordinates in the full pretif space.
    Units are voxels.
    """

    def __init__(
        self,
        data_paths=None,  # Union[str, List[str]]
        json_paths=None,  # Union[str, List[str]]
        point2segment=False,
        mixture_weights=None,  # Optional[Union[float, List[float]]]
        dataset_size=512,
        max_vol_num=5,
        subvol_size=(32, 256, 256),
        box_width=16,
        transforms=None,
        params=None,
        labeled=True,
        im_channels=1,
        R=2,
        selected_axon_id=None,
        instance=True,
        twochannel=False,
    ) -> None:
        super().__init__()

        self.num_datasets = len(data_paths)
        if mixture_weights is None:
            mixture_weights = [1.0] * self.num_datasets
        if not (len(data_paths) == len(json_paths) == len(mixture_weights)):
            raise ValueError(
                "data_paths, json_paths, and mixture_weights must have the same length"
            )

        self.point2segment = point2segment
        weight_arr = np.asarray(mixture_weights, dtype=float)
        self.mixture_weights = (weight_arr / weight_arr.sum()).tolist()

        self.data_paths = [Path(p) for p in data_paths]
        self.json_paths = [Path(p) for p in json_paths]

        self.subvol_size = np.array(subvol_size).astype(int)
        self.max_vol_num = max_vol_num
        self.box_width = box_width
        self.labeled = labeled
        self.transforms = transforms or []
        self.params = params or {}
        self.crop_vol_size = (
            (self.subvol_size * params["resize_ratio"][1]).astype(int)
            if "resize_ratio" in params
            else self.subvol_size
        )
        self.im_channels = im_channels
        self.R = R
        self.instance = instance
        self.twochannel = twochannel
        # print(f'self.crop_vol_size: {self.crop_vol_size}')

        # ------------------------------------------------------------------
        # Parse JSON → list of (brain_id, axon_key, swc_path, subvol_folder)
        # ------------------------------------------------------------------
        self.samples = []
        self.data_vols = []
        self.sampling_weights = []

        for dataset_idx, (weight, data_path, json_path) in enumerate(
            zip(self.mixture_weights, self.data_paths, self.json_paths)
        ):
            json_file = Path(json_path)
            if json_file.suffix != ".json":
                json_file = json_file.with_suffix(".json")
            if not json_file.is_absolute():
                json_file = data_path / json_file

            with open(json_file) as f:
                brains = json.load(f)
            if not isinstance(brains, list):
                brains = [brains]

            current_data_vols = []
            for brain in brains:
                brain_id = brain["brain_ID"]
                modality = brain.get("modality", "confocal")

                if not self.labeled:
                    assert "range" in brain, f"brain {brain_id} does not have range"
                    assert "marker_folder" in brain, f"brain {brain_id} does not have marker_folder"

                    # TODO sliding window for range
                    for vrange in brain["range"]:
                        current_data_vols.append(
                            {
                                "brain_id": brain_id,
                                "marker_folder": brain["marker_folder"],
                                "range": vrange,
                                "dataset_idx": dataset_idx,
                                "data_path": data_path,
                                "modality": modality,
                            }
                        )
                else:
                    offset = brain["offset"]
                    ntype = brain["ntype"] if "ntype" in brain else (3, 4)
                    if modality == "fMOST":
                        # ntype = (4,)
                        if selected_axon_id is not None:
                            for axon_id in selected_axon_id:
                                axon_key = f"axon_of_{axon_id}"
                                swc_rel = brain["annotation_dict"][axon_key]
                                swc_path = data_path / brain_id / "1" / swc_rel
                                for subvol_folder in brain["subvol_dict"][axon_key]:
                                    current_data_vols.append(
                                        {
                                            "brain_id": brain_id,
                                            "axon_key": axon_key,
                                            "swc_path": swc_path,
                                            "subvol_folder": subvol_folder,  # if "pretif" in subvol_folder else f"{subvol_folder}_pretif",
                                            "offset": offset,
                                            "ntype": ntype,
                                            "dataset_idx": dataset_idx,
                                            "data_path": data_path,
                                            "modality": modality,
                                        }
                                    )
                        else:
                            for axon_key, swc_rel in brain["annotation_dict"].items():
                                swc_path = data_path / brain_id / "1" / swc_rel
                                for subvol_folder in brain["subvol_dict"][axon_key]:
                                    current_data_vols.append(
                                        {
                                            "brain_id": brain_id,
                                            "axon_key": axon_key,
                                            "swc_path": swc_path,
                                            "subvol_folder": subvol_folder,  # if "pretif" in subvol_folder else f"{subvol_folder}_pretif",
                                            "offset": offset,
                                            "ntype": ntype,
                                            "dataset_idx": dataset_idx,
                                            "data_path": data_path,
                                            "modality": modality,
                                        }
                                    )
                    else:
                        # for in-house data
                        current_data_vols.append(
                            {
                                "brain_id": brain_id,
                                "subvol_folder": brain["marker_folder"],
                                "range": brain[
                                    "range"
                                ],  # Note the boundary of the annotation #TODO
                                "axons": brain[
                                    "axons"
                                ],  # list of dict of list, "starting points": List[np.ndarray]
                                "offset": offset,
                                "ntype": (999),
                                "dataset_idx": dataset_idx,
                                "data_path": data_path,
                                "modality": modality,
                            }
                        )
            self.sampling_weights.extend([weight for _ in range(len(current_data_vols))])
            self.data_vols.extend(current_data_vols)

        # expand samples random sample from self.data_vols to have 512 samples
        if dataset_size > len(self.data_vols):
            for i in range(dataset_size):
                sample = random.choices(self.data_vols, weights=self.sampling_weights, k=1)[0]
                self.samples.append(sample)
        else:
            self.samples = self.data_vols

        # ------------------------------------------------------------------
        # pretif cache key = (brain_id, axon, subvol_folder)
        # ------------------------------------------------------------------
        self._tif_cache = {}
        for s in self.data_vols:
            modality = s.get("modality", "confocal")
            if self.labeled:
                if modality == "fMOST":
                    key = (s["brain_id"], s["axon_key"], s["subvol_folder"])
                    if key not in self._tif_cache:
                        if "pretif" in s["subvol_folder"]:
                            self._tif_cache[key] = pretif_open(
                                s["data_path"] / s["brain_id"] / s["axon_key"] / s["subvol_folder"]
                            )
                        elif ".tif" in s["subvol_folder"]:
                            self._tif_cache[key] = tifffile.imread(
                                s["data_path"] / s["brain_id"] / s["subvol_folder"]
                            )
                else:
                    subvol_range = "_".join(map(str, s["range"]))
                    key = (s["brain_id"], s["subvol_folder"], subvol_range)
                    if key not in self._tif_cache:
                        self._tif_cache[key] = pretif_open(
                            s["data_path"] / s["brain_id"] / s["subvol_folder"]
                        )[
                            s["range"][0][0] : s["range"][1][0],
                            s["range"][0][1] : s["range"][1][1],
                            s["range"][0][2] : s["range"][1][2],
                        ]
            else:
                key = (s["brain_id"], s["marker_folder"])
                if key not in self._tif_cache:
                    self._tif_cache[key] = pretif_open(
                        s["data_path"] / s["brain_id"] / s["marker_folder"]
                    )

    # ------------------------------------------------------------------
    # helper functions
    # ------------------------------------------------------------------
    @staticmethod
    def _point_in_crop(point, depth, height, width):
        z, y, x = point
        return 0.0 <= z <= depth - 1 and 0.0 <= y <= height - 1 and 0.0 <= x <= width - 1

    def _clip_neurite_to_crop(self, points, crop_shape):
        depth, height, width = crop_shape
        if not points:
            return []

        clipped_points = []
        prev = np.asarray(points[0], dtype=float)
        prev_inside = self._point_in_crop(prev, depth, height, width)
        if prev_inside:
            clipped_points.append(prev)

        for curr_raw in points[1:]:
            curr = np.asarray(curr_raw, dtype=float)
            curr_inside = self._point_in_crop(curr, depth, height, width)
            if prev_inside and curr_inside:
                clipped_points.append(curr)
            else:
                clipped = clip_segment_to_view(prev, curr, depth, height, width)
                if clipped is None:
                    prev, prev_inside = curr, curr_inside
                    continue
                start_clip, end_clip = clipped
                start_clip = np.array(start_clip, dtype=float)
                end_clip = np.array(end_clip, dtype=float)
                if prev_inside and not curr_inside:
                    clipped_points.append(end_clip)
                elif (not prev_inside) and curr_inside:
                    clipped_points.append(start_clip)
                    clipped_points.append(curr)
                else:
                    clipped_points.append(start_clip)
                    clipped_points.append(end_clip)
            prev, prev_inside = curr, curr_inside

        return clipped_points

    def _update_dense_targets(self, heatmap, instance_map, points, instance_id):
        if points is None or len(points) == 0:
            return
        local_heatmap = center_to_gaussian(
            points,
            subvol_size=self.subvol_size,
            R=self.R,
            object_radius=2,  # 4 for center, and axon before 0221, but 4 is too large for 4x
        )
        np.maximum(heatmap, local_heatmap, out=heatmap)
        instance_map[local_heatmap > 0.5] = instance_id

    # @staticmethod
    # def _connect_neurite_points(points):
    #     pts = np.asarray(points, dtype=float)
    #     if pts.ndim != 2 or pts.shape[0] < 2:
    #         return pts
    #     deltas = np.diff(pts, axis=0)
    #     steps = np.max(np.abs(deltas), axis=1).astype(int)
    #     if np.all(steps <= 1):
    #         return pts
    #     segments = []
    #     for i, step in enumerate(steps):
    #         if step <= 0:
    #             if not segments:
    #                 segments.append(pts[i:i + 1])
    #             continue
    #         t = np.linspace(0.0, 1.0, step + 1, dtype=float)
    #         seg = pts[i] + deltas[i] * t[:, None]
    #         if segments:
    #             seg = seg[1:]
    #         segments.append(seg)
    #     if not segments:
    #         return pts[:1]
    #     return np.concatenate(segments, axis=0)

    def random_sample(self, data_vol, raw_neurites_dict):
        # random sample z, y, x
        Z, Y, X = data_vol.shape
        dz, dy, dx = self.crop_vol_size
        all_points = []
        empty = False
        for points in raw_neurites_dict.values():
            all_points += points.tolist()
        # print(f'Among {len(raw_neurites_dict.values())}, find {len(all_points)} points')
        cover_points = False
        iterate = 0
        while not cover_points and iterate < 10 and not empty:
            # z = np.random.randint(0, data_size[0]-self.crop_vol_size[0])
            # y = np.random.randint(0, data_size[1]-self.crop_vol_size[1])
            # x = np.random.randint(0, data_size[2]-self.crop_vol_size[2])
            if len(all_points) < 1:
                empty = True
                z = np.random.randint(0, Z - dz)
                y = np.random.randint(0, Y - dy)
                x = np.random.randint(0, X - dx)
                print("could not find any points in the subvol.. random crop.")
                break
            # print(f'sample from {len(all_points)} points. first point: {all_points[0]}')
            # Randomly choose a positive point
            cz, cy, cx = all_points[np.random.randint(len(all_points))]

            # Compute valid ranges for the start indices to ensure the patch stays in bounds
            min_z = max(0, cz - dz + 1)
            max_z = min(cz, Z - dz)
            min_y = max(0, cy - dy + 1)
            max_y = min(cy, Y - dy)
            min_x = max(0, cx - dx + 1)
            max_x = min(cx, X - dx)

            # If the min and max are equal, then there's only one valid start index
            z = np.random.randint(min_z, max_z + 1) if max_z >= min_z else max_z
            y = np.random.randint(min_y, max_y + 1) if max_y >= min_y else max_y
            x = np.random.randint(min_x, max_x + 1) if max_x >= min_x else max_x
            if not np.isfinite([z, y, x]).all():
                z = np.random.randint(0, Z - dz)
                y = np.random.randint(0, Y - dy)
                x = np.random.randint(0, X - dx)
            z = int(np.clip(z, 0, Z - dz))
            y = int(np.clip(y, 0, Y - dy))
            x = int(np.clip(x, 0, X - dx))

            inbound_idx = np.nonzero(
                np.all(
                    ((np.array(all_points) - np.array([z, y, x])) < self.crop_vol_size)
                    * ((np.array(all_points) - np.array([z, y, x])) >= np.array([0, 0, 0])),
                    axis=1,
                )
            )[0].tolist()
            if len(inbound_idx) > 5:
                cover_points = True
            iterate += 1
        crop_vol = data_vol[
            z : z + self.crop_vol_size[0],
            y : y + self.crop_vol_size[1],
            x : x + self.crop_vol_size[2],
        ]
        origin = np.array([z, y, x], dtype=float)
        crop_shape = tuple(int(dim) for dim in self.crop_vol_size)
        for k, v in raw_neurites_dict.items():
            shifted = [np.asarray(p, dtype=float) - origin for p in v]
            raw_neurites_dict[k] = self._clip_neurite_to_crop(shifted, crop_shape)
        return np.expand_dims(crop_vol, axis=0), raw_neurites_dict, z, y, x, empty

    def fetch_points(
        self,
        neuron,
        zyx_min=(0, 0, 0),
        subvol_size=(400, 1000, 1000),
        annotation_offset=(10, 0, 0),
        ntype=(3, 4),
    ):
        # fetch for the subvol, and then random crop
        offset = np.array(zyx_min, dtype=float) - np.array(annotation_offset)
        max_bound = np.array(subvol_size, dtype=float)
        neurites_dict = {}
        neurites_idx = 0
        total_points = 0
        for section in iter_sections(neuron.neurites):
            # only care axon
            if section.type not in ntype:
                continue
            points = np.array(section.points[:, 2::-1]) - offset
            points = points[np.all((points >= 0) & (points < max_bound), axis=1)]
            if points.shape[0] < 2:
                continue
            neurites_dict[neurites_idx] = points
            total_points += points.shape[0]
            neurites_idx += 1
        # print(f'total points found: {total_points}')
        return neurites_dict

    def fetch_points_swctxt(
        self, swc_data, zyx_min=(0, 0, 0), annotation_offset=(10, 0, 0), merge_children=False
    ):
        offset = np.array(zyx_min, dtype=float) - np.array(annotation_offset)
        if swc_data.size == 0:
            return {}

        node_ids = swc_data[:, 0].astype(int)
        parent_ids = swc_data[:, 6].astype(int)
        id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

        children = {node_id: [] for node_id in node_ids}
        for node_id, parent_id in zip(node_ids, parent_ids):
            if parent_id == -1:
                continue
            if parent_id in children:
                children[parent_id].append(node_id)

        for node_id in children:
            children[node_id].sort()

        def point_of(node_id):
            idx = id_to_index[node_id]
            # swc columns: id, type, x, y, z, radius, parent; convert to z, y, x
            return swc_data[idx, 4:1:-1].astype(float) - offset

        segment_dict = {}
        segment_id = 0

        def is_start(node_id):
            parent_id = parent_ids[id_to_index[node_id]]
            if parent_id == -1:
                return True
            parent_children = children.get(parent_id, [])
            return len(parent_children) > 1

        for node_id in node_ids:
            if not is_start(node_id):
                continue
            for child_id in children.get(node_id, []):
                points = [point_of(node_id), point_of(child_id)]
                cur = child_id
                while True:
                    cur_children = children.get(cur, [])
                    if len(cur_children) != 1:
                        break
                    nxt = cur_children[0]
                    points.append(point_of(nxt))
                    cur = nxt
                segment_dict[segment_id] = np.array(points)
                segment_id += 1

        return segment_dict

    def fetch_points_from_list(
        self,
        axons_raw,
        zyx_min=(0, 0, 0),
        subvol_size=(400, 1000, 1000),
        annotation_offset=(10, 0, 0),
        ntype=(3, 4),
    ):
        offset = np.array(zyx_min, dtype=float) - np.array(annotation_offset)
        max_bound = np.array(subvol_size, dtype=float)
        neurites_dict = {}
        neurites_idx = 0
        total_points = 0
        for axon in axons_raw:
            for key, points in axon.items():
                points = np.array(points) - offset
                points = points[np.all((points >= 0) & (points < max_bound), axis=1)]
                if points.shape[0] < 2:
                    continue
                neurites_dict[neurites_idx] = points
                total_points += points.shape[0]
                neurites_idx += 1
        return neurites_dict

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample_flag = True
        while sample_flag:
            vol_num = random.randint(1, self.max_vol_num)
            vol_idxs = random.sample(range(len(self.samples)), vol_num)

            heatmap_size = tuple(self.subvol_size // self.R)
            heatmap = np.zeros(heatmap_size, dtype=np.float32)
            instance_map = np.full(heatmap_size, -1, dtype=np.int64)
            gt_neurites = []  # list of neurites in all volumes [Dict[List[np.ndarray]]]
            vol = np.zeros((self.im_channels, *self.subvol_size))
            instance_idx = 0
            # unlabeled data -> test data -> no vol overlay
            raw_neurites_dict = {}
            is_sparse = False
            if not self.labeled:
                sample_flag = False
                info = self.samples[i]
                brain_id, marker_folder = info["brain_id"], info["marker_folder"]
                (x_min, y_min, z_min), (x_max, y_max, z_max) = info["range"]
                # crop volume by range
                data_vol = self._tif_cache[(brain_id, marker_folder)]
                assert (
                    np.array([z_max, y_max, x_max])
                    - np.array([z_min, y_min, x_min])
                    - np.array(self.subvol_size)
                ).all() == 0, f"Invalid range {info['range']}"
                raw_vol = np.expand_dims(data_vol[z_min:z_max, y_min:y_max, x_min:x_max], axis=0)
                augmented = Transforms(
                    np.array(raw_vol).astype(np.float32), [], self.transforms, size=self.subvol_size
                )  # , range)
                augmented.get_params(self.params)
                augmented.apply_transform()
                vol = augmented.volume
            else:
                # overlay different volume: max proj
                for i, vol_idx in enumerate(vol_idxs):
                    info = self.samples[vol_idx]
                    if info["modality"] == "fMOST":
                        brain_id, axon_key, swc_path, subvol_folder, offset, ntype = (
                            info["brain_id"],
                            info["axon_key"],
                            info["swc_path"],
                            info["subvol_folder"],
                            info["offset"][::-1],
                            info["ntype"],
                        )

                        # load pretif volume (single‑channel); add channel dim
                        data_vol = self._tif_cache[(brain_id, axon_key, subvol_folder)]
                        data_vol_size = data_vol.shape
                        if "pretif" in subvol_folder:
                            x_min, _, y_min, _, z_min, _ = [
                                int(s) for s in subvol_folder.split("_") if s.isdigit()
                            ]  # absolute xyz min of this sub‑vol
                            # neuron = load_morphology(swc_path)
                            neuron = _safe_load_morphology(swc_path)
                            raw_neurites_dict = self.fetch_points(
                                neuron, (z_min, y_min, x_min), data_vol_size, offset, ntype
                            )  # List[List[np.ndarray]]
                        elif ".tif" in subvol_folder:
                            x_min, y_min, z_min = 0, 0, 0
                            swc_data = np.loadtxt(swc_path, ndmin=2)
                            raw_neurites_dict = self.fetch_points_swctxt(
                                swc_data, (z_min, y_min, x_min), offset
                            )
                        data_vol_size = data_vol.shape
                    else:
                        is_sparse = True
                        brain_id, subvol_folder, subvol_range, offset, ntype = (
                            info["brain_id"],
                            info["subvol_folder"],
                            info["range"],
                            info["offset"][::-1],
                            info["ntype"],
                        )

                        # load pretif volume (single‑channel); add channel dim
                        data_vol = self._tif_cache[
                            (brain_id, subvol_folder, "_".join(map(str, subvol_range)))
                        ]
                        z_min, y_min, x_min = subvol_range[0]
                        data_vol_size = data_vol.shape

                        neurons = info["axons"]
                        raw_neurites_dict = self.fetch_points_from_list(
                            neurons, (z_min, y_min, x_min), data_vol_size, offset, ntype
                        )  # List[List[np.ndarray]]

                    if np.any(np.array(data_vol_size) < self.crop_vol_size):
                        print(
                            f"data vol size: {data_vol_size} is smaller than crop vol size: {self.crop_vol_size}, skip this sample"
                        )
                        continue
                    raw_vol, raw_neurites_dict, z, y, x, empty = self.random_sample(
                        data_vol, raw_neurites_dict
                    )
                    if empty:
                        print(f"no points in {subvol_folder}, random crop")
                    augmented = Transforms(
                        np.array(raw_vol).astype(np.float32),
                        raw_neurites_dict,
                        self.transforms,
                        size=self.subvol_size,
                    )  # , range)
                    augmented.get_params(self.params)
                    augmented.apply_transform()

                    vol = np.maximum(vol, augmented.volume)
                    sample_flag = False
                    augmented_neurites = augmented.coords
                    gt_neurites.append(augmented_neurites)
                    num_points = 0

                    for points in augmented_neurites.values():
                        if points.shape[0] < 1:
                            continue
                        if self.point2segment:
                            points = connect_neurite_points(points)
                        num_points += points.shape[0]
                        self._update_dense_targets(heatmap, instance_map, points, instance_idx)
                        instance_idx += 1

        if self.instance:
            target = {
                "heatmap": torch.tensor(heatmap, dtype=torch.float32),
                "instance_map": torch.tensor(instance_map, dtype=torch.long),
                "is_sparse": is_sparse,
                "annotations": {"gt_neurites": gt_neurites},
            }
        else:
            target = {
                "heatmap": torch.tensor(heatmap, dtype=torch.float32),
            }

        if self.twochannel:
            vol = np.concatenate([np.zeros_like(vol), vol], axis=0)
        return torch.tensor(vol, dtype=torch.float32), target, torch.zeros(1)


def collate_detr(batch):
    voxels, targets, dataset_ids = zip(*batch)  # tuples
    voxels = torch.stack(voxels, 0)  # [B,C,D,H,W]
    dataset_ids = torch.stack(dataset_ids, 0)  # [B]
    return voxels, list(targets), dataset_ids
