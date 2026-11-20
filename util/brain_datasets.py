import os

import numpy as np
from torch.utils.data import Dataset
import torch
from .bmae_utils import pretif_open


class BrainImageDataset(Dataset):
    def __init__(
        self,
        data_path,
        data_folder_list,
        num_of_data=100000,
        min_coord=(0, 0, 0),
        max_coord=(3762, 9000, 7422),
        subvol_size=(128, 128, 128),
        min_percentile=1,
        max_percentile=99,
        min_intensity=150,
    ):
        super().__init__()
        self.data = []  # store the starting coordinate of random subvolume
        self.pretif_list = []
        for data_folder in data_folder_list:
            if data_folder:
                self.pretif_list.append(pretif_open(os.path.join(data_path, data_folder)))
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.subvol_size = subvol_size
        self.num_of_data = num_of_data

        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.min_intensity = min_intensity

    def __len__(self):
        return self.num_of_data

    def __getitem__(self, i):

        z, y, x = self.random_sample()
        data_list = []
        for pretif in self.pretif_list:
            data = pretif[
                z : z + self.subvol_size[0],
                y : y + self.subvol_size[1],
                x : x + self.subvol_size[2],
            ]
            # normalization
            min = np.percentile(data, self.min_percentile)
            max = np.percentile(data, self.max_percentile)
            data = (data - min) / (max - min + 1e-6)
            data = np.clip(data, 0, 1)
            data_list.append(data)

        data_list = torch.Tensor(np.array(data_list).astype(np.float32))
        return data_list

    def random_sample(self):
        # random sample z, y, x
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
                    z : z + self.subvol_size[0],
                    y : y + self.subvol_size[1],
                    x : x + self.subvol_size[2],
                ]
            )
        return z, y, x
