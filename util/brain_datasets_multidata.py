import os

import numpy as np
from torch.utils.data import Dataset
import torch
from .bmae_utils import pretif_open


class BrainImageDataset(Dataset):
    def __init__(
        self,
        data_paths,
        data_folder_lists,
        num_of_data=100000,
        min_coord=(10, 0, 0),
        max_coord=(300, 10000, 12400),
        subvol_size=(32, 256, 256),
        min_percentile=1,
        max_percentile=99,
        min_intensity=110,
    ):
        super().__init__()
        if isinstance(data_paths, (str, os.PathLike)):
            data_paths = [data_paths]
        if not data_paths:
            raise ValueError("At least one pretraining data path is required")
        folders_by_channel = []
        for folders in data_folder_lists:
            if not folders:
                continue
            if isinstance(folders, str):
                folders = [folders]
            if len(folders) != len(data_paths):
                raise ValueError("Each channel needs one folder name per data path")
            folders_by_channel.append(folders)
        if not folders_by_channel:
            raise ValueError("At least one pretraining image channel is required")
        self.data = []  # store the starting coordinate of random subvolume
        self.pretif_list = {}
        for i, data_path in enumerate(data_paths):
            pretifs = []
            for data_folders in folders_by_channel:
                pretifs.append(pretif_open(os.path.join(data_path, data_folders[i])))
            self.pretif_list[i] = pretifs
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
        i = np.random.randint(0, len(self.pretif_list))
        z, y, x = self.random_sample(i)
        data_list = []
        pretifs = self.pretif_list[i]
        for pretif in pretifs:
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
        # class_idx = torch.tensor(i+1, dtype=torch.float32) # for cell, differentiate brain volumes: necessary?
        class_idx = torch.tensor(0, dtype=torch.float32)  # for axon, all set to zero
        return data_list, class_idx

    def random_sample(self, i):
        # random sample z, y, x
        max_voxel = 0
        if len(self.pretif_list[i]) == 1:
            pretif = self.pretif_list[i][0]
        else:
            pretif = self.pretif_list[i][1]

        pretif_size = np.array(pretif.shape) - np.array([5, 50, 50])
        max_coord = np.maximum(pretif_size, self.max_coord)
        while max_voxel < self.min_intensity:
            z = np.random.randint(self.min_coord[0], max_coord[0])
            y = np.random.randint(self.min_coord[1], max_coord[1])
            x = np.random.randint(self.min_coord[2], max_coord[2])
            max_voxel = np.max(
                pretif[
                    z : z + self.subvol_size[0],
                    y : y + self.subvol_size[1],
                    x : x + self.subvol_size[2],
                ]
            )
        return z, y, x
