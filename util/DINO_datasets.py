import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .bmae_utils import pretif_open
from .transforms import Transforms


class BrainDINOImageDataset(Dataset):
    """Sample a BrainMAE-style 128^3 parent volume and return DINO multi-crops."""

    def __init__(
        self,
        data_path,
        data_folder_list,
        num_of_data=100000,
        min_coord=(0, 0, 0),
        max_coord=(3762, 9000, 7422),
        parent_size=(128, 128, 128),
        global_size=(96, 96, 96),
        local_size=(64, 64, 64),
        local_crops_number=8,
        transforms=("resize", "flip", "gamma"),
        params=None,
        min_intensity=150,
    ):
        super().__init__()
        self.pretif_list = []
        for data_folder in data_folder_list:
            if data_folder:
                self.pretif_list.append(pretif_open(os.path.join(data_path, data_folder)))

        self.min_coord = min_coord
        self.max_coord = max_coord
        self.parent_size = tuple(parent_size)
        self.global_size = tuple(global_size)
        self.local_size = tuple(local_size)
        self.local_crops_number = local_crops_number
        self.num_of_data = num_of_data
        self.transforms = list(transforms)
        self.params = params or {
            "resize_ratio": (0.8, 1.2),
            "resize_mode": "trilinear",
            "gamma_range": (0.8, 2.0),
            "flip_axes": [-1, -2, -3],
            "norm_percentiles": (1, 99),
        }
        self.min_intensity = min_intensity

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, _):
        z, y, x = self.random_sample()
        parent = []
        for pretif in self.pretif_list:
            data = pretif[
                z : z + self.parent_size[0],
                y : y + self.parent_size[1],
                x : x + self.parent_size[2],
            ]
            parent.append(data)
        parent = np.array(parent).astype(np.float32)

        crops = []
        for _ in range(2):
            crops.append(self._make_view(parent, self.global_size))
        for _ in range(self.local_crops_number):
            crops.append(self._make_view(parent, self.local_size))
        return crops

    def _make_view(self, parent, crop_size):
        # Transforms samples resize/gamma/flip parameters independently per call.
        transform = Transforms(
            parent.copy(), coords=None, transforms=self.transforms, size=crop_size
        )
        transform.get_params(self.params)
        transform.apply_transform()
        return torch.from_numpy(transform.volume.astype(np.float32).copy())

    def random_sample(self):
        max_voxel = 0
        if len(self.pretif_list) == 1:
            pretif = self.pretif_list[0]
        else:
            pretif = self.pretif_list[1]
        while max_voxel < self.min_intensity:
            z = np.random.randint(self.min_coord[0], self.max_coord[0])
            y = np.random.randint(self.min_coord[1], self.max_coord[1])
            x = np.random.randint(self.min_coord[2], self.max_coord[2])
            max_voxel = np.max(
                pretif[
                    z : z + self.parent_size[0],
                    y : y + self.parent_size[1],
                    x : x + self.parent_size[2],
                ]
            )
        return z, y, x


BrainDINOMultiCropDataset = BrainDINOImageDataset
