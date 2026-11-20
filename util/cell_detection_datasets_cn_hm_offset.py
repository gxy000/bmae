import os
import numpy as np
from torch.utils.data import Dataset
import torch
import json
import math
import random
from .bmae_utils import pretif_open
from .transforms import Transforms
from skimage.feature import peak_local_max

from scipy.spatial import cKDTree


import logging

dynamics_logger = logging.getLogger(__name__)


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
        # TODO: check radius
        heatmap = draw_umich_gaussian(heatmap, point // R, radius=object_radius // R)

    return heatmap


def center_to_offset(points, subvol_size, R):
    subvol_size = tuple(np.array(subvol_size) // R)
    points = points / float(R)
    tree = cKDTree(points)
    # Generate all voxel coordinates in the volume
    z, y, x = np.meshgrid(
        np.arange(subvol_size[0]),
        np.arange(subvol_size[1]),
        np.arange(subvol_size[2]),
        indexing="ij",
    )
    voxel_grid = np.stack([z, y, x], axis=-1).reshape(-1, 3)
    distances, idxs = tree.query(voxel_grid)
    offsets = voxel_grid - points[idxs]
    voxel_offset = offsets.reshape(subvol_size + (3,))
    voxel_offset = voxel_offset.transpose(-1, 0, 1, 2)

    return voxel_offset


def _cn_decode_one_cls(heat, reg=None, min_distance=1, threshold_heat=0.5, exclude_border=True):
    if isinstance(heat, torch.Tensor):
        heat = heat.squeeze(0).detach().cpu().numpy()

    centers = peak_local_max(
        heat[0, :, :, :],
        min_distance=min_distance,
        threshold_abs=threshold_heat,
        exclude_border=exclude_border,
    )
    zs, ys, xs = centers[:, 0], centers[:, 1], centers[:, 2]
    scores = heat[:, zs, ys, xs].reshape(-1, 1)
    if reg is not None:
        if isinstance(reg, torch.Tensor):
            reg = reg.squeeze(0).detach().cpu().numpy()
        reg = reg.transpose(1, 2, 3, 0)
        centers = centers + reg[zs, ys, xs, :]
    else:
        centers = centers + np.array([0.5, 0.5, 0.5])

    return np.concatenate([centers, scores], axis=-1)


# @njit("(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], uint32)", nogil=True)


class CellDetectionDatset(Dataset):
    def __init__(
        self,
        data_path,
        json_path,
        subvol_size=(128, 128, 128),
        transforms=[],
        params={},
        box_width=16,
        batched=True,
        marker_only=False,
        two_channel=True,
        R=2,
        labeled=True,
        dataset_size=0,
    ):
        """
        transform: "resize", "flip", "gamma"
        params:
            "resize_ratio": (0.8, 1.2)
            "resize_mode": "trilinear", "nearest"
            "gamma_range": (0.9, 1.1)
            "flip_axes": -1, -2, -3
            "norm_percentiles": (1, 99)
        """
        super().__init__()

        if ".jsonl" in json_path:
            with open(json_path) as f:
                self.data = [json.loads(line) for line in f]
        else:
            with open(json_path) as f:
                self.data = json.load(f)["annotations"]
        self.subvol_size = subvol_size
        self.transforms = transforms
        self.params = params
        self.box_width = box_width
        self.batched = batched
        self.marker_only = marker_only
        self.two_channel = two_channel
        self.R = R
        self.labeled = labeled
        self.dataset_size = int(dataset_size)

        self.pretif_dict = {}
        for annotation in self.data:
            #         self.pretif_dict[path] = pretif_open(os.path.join(data_path, path))
            for i, data_folder in enumerate(annotation["channels"]):
                if i == 0 and self.marker_only:
                    continue
                path = os.path.join(annotation["brain_id"], data_folder)
                if path not in self.pretif_dict:
                    self.pretif_dict[path] = pretif_open(os.path.join(data_path, path))

        if self.dataset_size < 0:
            raise ValueError("dataset_size must be >= 0")
        if self.dataset_size > 0:
            if len(self.data) == 0:
                raise ValueError("Cannot sample from an empty dataset when dataset_size > 0")
            self.samples = random.choices(self.data, k=self.dataset_size)
        else:
            self.samples = self.data

    def __len__(self):
        if self.dataset_size == 0:
            return len(self.data)
        return len(self.samples)

    def __getitem__(self, i):
        """
        return:
        data_list: list of 3D numpy array, random crop to 128x128x128 at augmentation time
        centers: annotated center of the cell in the cropped image, numpy array of [M, 3]
        """
        if self.dataset_size == 0:
            annotation = self.data[i]
        else:
            annotation = self.samples[i]

        data_list = []

        [zmin, ymin, xmin], [zmax, ymax, xmax] = annotation["range"]
        extend = 0
        targets = np.array(annotation["centers"]).astype(int)

        if self.labeled:
            if "rotate" in self.params:
                extend = math.ceil(((2) ** 0.5 - 1) * max(ymax - ymin, xmax - xmin) / 2)
                ymin, xmin = ymin - extend, xmin - extend
                ymax, xmax = ymax + extend, xmax + extend
        if targets.shape[0] > 0:
            targets[..., :] -= np.array([zmin, ymin, xmin])

        # read in image volumes
        for i, data_folder in enumerate(annotation["channels"]):
            if i == 0 and self.marker_only:
                if self.two_channel:
                    data_list.append(np.zeros((zmax - zmin, ymax - ymin, xmax - xmin)))
                continue
            path = os.path.join(annotation["brain_id"], data_folder)
            data_list.append(self.pretif_dict[path][zmin:zmax, ymin:ymax, xmin:xmax])

        augmented = Transforms(
            np.array(data_list).astype(np.float32), targets, self.transforms
        )  # , range)
        augmented.get_params(self.params)
        augmented.apply_transform()
        data_list = torch.Tensor(augmented.volume.copy())
        targets = augmented.coords

        if self.labeled:
            heatmap = center_to_gaussian(
                targets, subvol_size=self.subvol_size, R=self.R, object_radius=self.box_width // 2
            )
            if not targets.size:
                offset = np.zeros((3, *heatmap.shape))
            else:
                offset = center_to_offset(targets, subvol_size=self.subvol_size, R=self.R)
            heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()
            offset = torch.from_numpy(offset)

            if self.batched:
                return data_list, heatmap, offset
            else:
                return data_list, heatmap, offset, targets

        else:
            start_coords = torch.tensor([zmin, ymin, xmin], dtype=torch.int16)
            return data_list, start_coords, targets
