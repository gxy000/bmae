import numpy as np
import torch
import random
import torch.nn.functional as F
import math
import torchvision.transforms as T


def normalize_by_percentile(volume, min_percentile, max_percentile):
    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()
    if len(volume.shape) == 3:
        volume = np.expand_dims(volume, axis=0)
    for channel in range(volume.shape[0]):
        data = volume[channel]
        min = np.percentile(data, min_percentile)
        max = np.percentile(data, max_percentile)
        data = (data - min) / (max - min + 1e-6)
        volume[channel] = np.clip(data, 0, 1)
    return volume


class Transforms:
    def __init__(self, volume, coords, transforms, size=(128, 128, 128)):
        self.volume = volume

        if isinstance(coords, dict):
            self.coords, self.coords_dict, self.coords_inbound = self.preprocess_coords_dict(coords)
            self.coords_is_dict = True
        else:
            self.coords = self._coerce_coords_array(coords)
            if self.coords is None:
                self.coords_inbound = {}
            else:
                self.coords_inbound = {i: 0 for i in range(len(self.coords))}
            self.coords_is_dict = False

        if isinstance(size, int):
            self.size = (size, size, size)
        else:
            self.size = size

        self.transforms = transforms
        self.params = {}

    @staticmethod
    def _coerce_coords_array(coords):
        if coords is None:
            return None

        coords = np.asarray(coords)
        if coords.size == 0:
            return np.empty((0, 3), dtype=float)
        if coords.ndim == 1:
            if coords.shape[0] != 3:
                raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
            return coords.reshape(1, 3)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
        return coords

    @staticmethod
    def _point_in_crop(point, depth, height, width):
        z, y, x = point
        return 0.0 <= z <= depth - 1 and 0.0 <= y <= height - 1 and 0.0 <= x <= width - 1

    @staticmethod
    def _clip_segment_to_view(p1, p2, depth, height, width):
        """
        Liang-Barsky segment clipping in (z, y, x); returns (start, end) or None if fully outside.
        """
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

    def _clip_neurite_to_crop(self, points, crop_shape):
        """
        Clip a polyline (list/array of zyx points) to crop_shape; may insert boundary points.
        """
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
                clipped = self._clip_segment_to_view(prev, curr, depth, height, width)
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

    def get_params(self, params):
        if "resize_ratio" in params.keys():
            self.params["resize_ratio"] = (
                np.random.uniform(*params["resize_ratio"]),
                np.random.uniform(*params["resize_ratio"]),
                np.random.uniform(*params["resize_ratio"]),
            )
            if "resize_mode" in params.keys():
                self.params["resize_mode"] = params["resize_mode"]
            else:
                self.params["resize_mode"] = "trilinear"
        else:
            self.params["resize_ratio"] = (1, 1, 1)
            self.params["resize_mode"] = "trilinear"

        if "gamma_range" in params.keys():
            self.params["gamma"] = np.random.uniform(*params["gamma_range"])
        if "flip_axes" in params.keys():
            self.params["flip_axes"] = params["flip_axes"]

        if "rotate_degrees" in params.keys():
            self.params["rotate_degrees"] = np.random.uniform(*params["rotate_degrees"])
        else:
            self.params["rotate_degrees"] = np.random.uniform(0, 360)

        if "norm_values" in params.keys():
            self.params["norm_values"] = params["norm_values"]
        elif "norm_percentiles" in params.keys():
            self.params["norm_percentiles"] = params["norm_percentiles"]
        else:
            self.params["norm_percentiles"] = (1, 99)

    def apply_transform(self):
        # resize and crop to patch size
        # TODO: model Config
        resize_flag = True if "resize" in self.transforms else False
        rotate_flag = True if "rotate" in self.transforms else False
        self.resize_and_rotate(
            resize_ratio=self.params["resize_ratio"],
            method=self.params["resize_mode"],
            rotate_degrees=self.params["rotate_degrees"],
            resize=resize_flag,
            rotate=rotate_flag,
        )
        if rotate_flag:
            self.center_crop(axial=False)

        if "gamma" in self.transforms:
            self.volume = self.volume ** self.params["gamma"]
        if "normalize_by_value" in self.transforms:
            self.normalize_by_value(*self.params["norm_values"])
        else:
            self.normalize_by_percentile(*self.params["norm_percentiles"])

        if "flip" in self.transforms:
            self.random_flip(self.params["flip_axes"])
        if "center_crop" in self.transforms:
            self.center_crop()
        elif "upperleft_crop" in self.transforms:
            self.upperleft_crop()
        elif "mode_crop" in self.transforms:
            self.mode_crop()
        elif "whole_volume" in self.transforms:
            self.coords_inbound = {k: 1 for k in self.coords_inbound.keys()}
        else:
            self.random_crop()

        if self.coords_is_dict:
            self.coords = self.postprocess_coords_dict(self.coords, self.coords_dict)
        # Normalize

    def preprocess_coords_dict(self, coords):
        # when coords is {1: list_of_coords, 2: list_of_coords, 3: list_of_coords},
        # convert to numpy array
        # use a dict to store idx of array
        c_list = []
        c_dict = {}
        c_inbound_dict = {}
        p_idx = 0

        for k, v in coords.items():
            p_list = []
            for p in v:
                c_inbound_dict[p_idx] = 0
                p_list.append(p_idx)
                c_list.append(p)
                p_idx += 1
            c_dict[k] = p_list
        return self._coerce_coords_array(c_list), c_dict, c_inbound_dict

    def postprocess_coords_dict(self, coords, c_dict):
        coords = coords.tolist()
        out_dict = {}
        for k, v in c_dict.items():
            out_dict[k] = np.array([coords[i] for i in v if self.coords_inbound[i] == 1])
        return out_dict

    def resize_and_rotate(self, resize_ratio, method, rotate_degrees, resize=False, rotate=False):
        """
        resize_ratio: 3 elements tuple for Z, Y, X
        """

        target_size = tuple((self.volume.shape[-3:] * np.array(resize_ratio)).astype(int))

        volume = torch.from_numpy(self.volume)
        ndim = volume.ndim

        for _ in range(5 - ndim):
            volume = volume.unsqueeze(0)

        if resize:
            assert method in ("trilinear", "nearest")

            align_corners = None
            if method != "nearest":
                align_corners = True

            res_volume = F.interpolate(
                volume, size=target_size, align_corners=align_corners, mode=method
            )
            if self.coords is not None and len(self.coords) > 0:
                self.coords = (self.coords * np.array(resize_ratio)).astype(int)
        else:
            res_volume = volume

        if rotate:
            # rotate x, y only
            for i in range(res_volume.shape[-3]):
                res_volume[..., i, :, :] = T.functional.rotate(
                    res_volume[..., i, :, :],
                    rotate_degrees,
                    interpolation=T.InterpolationMode.BILINEAR,
                )
            self.rotate_coords(rotate_degrees)
        for _ in range(5 - ndim):
            res_volume = res_volume[0]

        self.volume = res_volume.numpy()

    def rotate_coords(self, degrees):
        angle = math.radians(degrees)

        center_y = self.volume.shape[-2] / 2
        center_x = self.volume.shape[-1] / 2

        if self.coords is None or len(self.coords) == 0:
            return

        # Rotation matrix multiplication to get rotated x & y
        yr = (
            ((self.coords[:, 1] - center_y) * math.cos(angle))
            - ((self.coords[:, 2] - center_x) * math.sin(angle))
            + center_y
        )
        xr = (
            ((self.coords[:, 1] - center_y) * math.sin(angle))
            + ((self.coords[:, 2] - center_x) * math.cos(angle))
            + center_x
        )

        self.coords[:, 1:] = np.stack([yr, xr], axis=-1)

    def random_crop(self):
        if self.coords_is_dict:
            in_chans = self.volume.shape[0]
            volume = self.volume
            coords = self.coords
            zmax, ymax, xmax = volume.shape[-3:]

            inbound_idx = []
            iter = 0
            z0, y0, x0 = 0, 0, 0
            z1, y1, x1 = self.size
            volume_pad = np.zeros(tuple([in_chans, *self.size]))

            while iter < 10:
                # avoid being stuck forever
                iter += 1
                coords_inbound = {k: 0 for k in self.coords_inbound.keys()}

                z0 = random.randint(0, max(0, zmax - self.size[0]))
                y0 = random.randint(0, max(0, ymax - self.size[1]))
                x0 = random.randint(0, max(0, xmax - self.size[2]))

                z1 = min(z0 + self.size[0], zmax)
                y1 = min(y0 + self.size[1], ymax)
                x1 = min(x0 + self.size[2], xmax)

                if coords.shape[0] == 0:
                    volume_crop = volume[:, z0:z1, y0:y1, x0:x1]
                    volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume_crop
                    self.volume = volume_pad
                    return

                inbound_idx = np.nonzero(
                    np.all(
                        (coords < np.array([z1, y1, x1])) * (coords >= np.array([z0, y0, x0])),
                        axis=1,
                    )
                )[0].tolist()
                for idx in inbound_idx:
                    coords_inbound[idx] = 1
                if self.coords is None or len(inbound_idx) > 0:
                    self.coords_inbound = coords_inbound
                    break
            volume_crop = volume[:, z0:z1, y0:y1, x0:x1]
            volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume_crop
            self.volume = volume_pad

            if self.coords is not None and len(self.coords) > 0:
                crop_start = np.array([z0, y0, x0], dtype=float)
                crop_shape = (z1 - z0, y1 - y0, x1 - x0)

                new_coords = []
                new_dict = {}
                new_inbound = {}
                new_idx = 0

                for k, v in self.coords_dict.items():
                    neurite_points = [self.coords[i] for i in v]
                    shifted = [np.asarray(p, dtype=float) - crop_start for p in neurite_points]
                    clipped = self._clip_neurite_to_crop(shifted, crop_shape)
                    if len(clipped) == 0:
                        new_dict[k] = []
                        continue
                    idx_list = []
                    for p in clipped:
                        new_coords.append(p)
                        new_inbound[new_idx] = 1
                        idx_list.append(new_idx)
                        new_idx += 1
                    new_dict[k] = idx_list

                if len(new_coords) == 0:
                    self.coords = np.empty((0, 3))
                else:
                    self.coords = np.array(new_coords, dtype=float)
                self.coords_dict = new_dict
                self.coords_inbound = new_inbound
        else:
            in_chans = self.volume.shape[0]
            volume = self.volume
            coords = self.coords
            zmax, ymax, xmax = volume.shape[-3:]

            coords_out = np.empty((0, 3))
            iter = 0
            volume_pad = np.zeros(tuple([in_chans,] + list(self.size)))

            while iter < 10:
                # avoid being stuck forever
                iter += 1

                z0 = random.randint(0, max(0, zmax - self.size[0]))
                y0 = random.randint(0, max(0, ymax - self.size[1]))
                x0 = random.randint(0, max(0, xmax - self.size[2]))

                z1 = min(z0 + self.size[0], zmax)
                y1 = min(y0 + self.size[1], ymax)
                x1 = min(x0 + self.size[2], xmax)

                volume_crop = volume[:, z0:z1, y0:y1, x0:x1]
                volume_pad = np.zeros(tuple([in_chans,] + list(self.size)))
                volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume_crop

                if coords is None or len(coords) == 0:
                    break
                inbound_idx = np.nonzero(
                    np.all(
                        (coords < np.array([z1, y1, x1])) * (coords >= np.array([z0, y0, x0])),
                        axis=1,
                    )
                )[0]
                coords_out = coords[inbound_idx] - np.array([z0, y0, x0])
                if coords_out.shape[0] > 0:
                    break

            self.volume = volume_pad
            self.coords = coords_out if coords is not None else None

    def center_crop(self, axial=True):
        in_chans = self.volume.shape[0]
        volume = self.volume
        zmax, ymax, xmax = volume.shape[-3:]

        y0 = (ymax - self.size[1]) // 2
        x0 = (xmax - self.size[2]) // 2

        y1 = min(y0 + self.size[1], ymax)
        x1 = min(x0 + self.size[2], xmax)
        if axial:
            z0 = (zmax - self.size[0]) // 2
            z1 = min(z0 + self.size[0], zmax)
            volume_pad = np.zeros(tuple([in_chans, *self.size]))
        else:
            z0, z1 = 0, zmax
            volume_pad = np.zeros(tuple([in_chans, (z1 - z0), *self.size[1:]]))

        volume_crop = volume[:, z0:z1, y0:y1, x0:x1]

        volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume_crop
        if self.coords is not None and len(self.coords) > 0:
            crop_start = np.array([z0, y0, x0], dtype=float)
            crop_shape = (z1 - z0, y1 - y0, x1 - x0)
            coords = self.coords

            if self.coords_is_dict:
                new_coords = []
                new_dict = {}
                new_inbound = {}
                new_idx = 0

                for k, v in self.coords_dict.items():
                    neurite_points = [coords[i] for i in v]
                    shifted = [np.asarray(p, dtype=float) - crop_start for p in neurite_points]
                    clipped = self._clip_neurite_to_crop(shifted, crop_shape)
                    if len(clipped) == 0:
                        new_dict[k] = []
                        continue
                    idx_list = []
                    for p in clipped:
                        new_coords.append(p)
                        new_inbound[new_idx] = 1
                        idx_list.append(new_idx)
                        new_idx += 1
                    new_dict[k] = idx_list

                if len(new_coords) == 0:
                    self.coords = np.empty((0, 3))
                else:
                    self.coords = np.array(new_coords, dtype=float)
                self.coords_dict = new_dict
                self.coords_inbound = new_inbound
            else:
                inbound_idx = np.nonzero(
                    np.all(
                        (coords < np.array([z1, y1, x1])) * (coords >= np.array([z0, y0, x0])),
                        axis=1,
                    )
                )[0].tolist()
                if not hasattr(self, "coords_inbound") or len(self.coords_inbound) != len(coords):
                    self.coords_inbound = {i: 0 for i in range(len(coords))}
                for idx in inbound_idx:
                    self.coords_inbound[idx] = 1

                coords_out = coords[inbound_idx]
                self.coords = coords_out - crop_start
        self.volume = volume_pad

    def upperleft_crop(self):
        if self.coords_is_dict:
            in_chans = self.volume.shape[0]
            volume = self.volume
            coords = self.coords
            zmax, ymax, xmax = volume.shape[-3:]

            z0 = y0 = x0 = 0
            z1 = min(z0 + self.size[0], zmax)
            y1 = min(y0 + self.size[1], ymax)
            x1 = min(x0 + self.size[2], xmax)

            volume_pad = np.zeros(tuple([in_chans, *self.size]))
            volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume[
                :, z0:z1, y0:y1, x0:x1
            ]

            if coords is None or len(coords) == 0:
                self.volume = volume_pad
                return

            crop_start = np.array([z0, y0, x0], dtype=float)
            crop_shape = (z1 - z0, y1 - y0, x1 - x0)

            new_coords = []
            new_dict = {}
            new_inbound = {}
            new_idx = 0

            for k, v in self.coords_dict.items():
                neurite_points = [coords[i] for i in v]
                shifted = [np.asarray(p, dtype=float) - crop_start for p in neurite_points]
                clipped = self._clip_neurite_to_crop(shifted, crop_shape)
                if len(clipped) == 0:
                    new_dict[k] = []
                    continue
                idx_list = []
                for p in clipped:
                    new_coords.append(p)
                    new_inbound[new_idx] = 1
                    idx_list.append(new_idx)
                    new_idx += 1
                new_dict[k] = idx_list

            if len(new_coords) == 0:
                self.coords = np.empty((0, 3))
            else:
                self.coords = np.array(new_coords, dtype=float)
            self.coords_dict = new_dict
            self.coords_inbound = new_inbound
            self.volume = volume_pad
        else:
            in_chans = self.volume.shape[0]
            volume = self.volume
            coords = self.coords
            zmax, ymax, xmax = volume.shape[-3:]

            z0 = y0 = x0 = 0
            z1 = min(z0 + self.size[0], zmax)
            y1 = min(y0 + self.size[1], ymax)
            x1 = min(x0 + self.size[2], xmax)

            volume_pad = np.zeros(tuple([in_chans, *self.size]))
            volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume[
                :, z0:z1, y0:y1, x0:x1
            ]

            if coords is not None:
                inbound_idx = np.nonzero(
                    np.all(
                        (coords < np.array([z1, y1, x1])) * (coords >= np.array([z0, y0, x0])),
                        axis=1,
                    )
                )[0]
                coords_out = coords[inbound_idx] - np.array([z0, y0, x0])
                self.coords = coords_out

            self.volume = volume_pad

    def mode_crop(self):
        def _mode_start(axis_coords, crop, limit):
            if axis_coords.size == 0 or limit == 0:
                return 0
            clipped = np.clip(axis_coords.astype(int), 0, max(0, limit - 1))
            mode_center = int(np.bincount(clipped, minlength=limit).argmax())
            start = mode_center - crop // 2
            return min(max(start, 0), max(0, limit - crop))

        in_chans = self.volume.shape[0]
        volume = self.volume
        coords = self.coords if self.coords is not None else np.empty((0, 3))
        zmax, ymax, xmax = volume.shape[-3:]

        z0 = _mode_start(coords[:, 0] if coords.size else np.array([]), self.size[0], zmax)
        y0 = _mode_start(coords[:, 1] if coords.size else np.array([]), self.size[1], ymax)
        x0 = _mode_start(coords[:, 2] if coords.size else np.array([]), self.size[2], xmax)

        z1 = min(z0 + self.size[0], zmax)
        y1 = min(y0 + self.size[1], ymax)
        x1 = min(x0 + self.size[2], xmax)

        volume_pad = np.zeros(tuple([in_chans, *self.size]))
        volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume[:, z0:z1, y0:y1, x0:x1]

        if self.coords_is_dict:
            if coords.size == 0:
                self.volume = volume_pad
                return

            crop_start = np.array([z0, y0, x0], dtype=float)
            crop_shape = (z1 - z0, y1 - y0, x1 - x0)

            new_coords = []
            new_dict = {}
            new_inbound = {}
            new_idx = 0

            for k, v in self.coords_dict.items():
                neurite_points = [coords[i] for i in v]
                shifted = [np.asarray(p, dtype=float) - crop_start for p in neurite_points]
                clipped = self._clip_neurite_to_crop(shifted, crop_shape)
                if len(clipped) == 0:
                    new_dict[k] = []
                    continue
                idx_list = []
                for p in clipped:
                    new_coords.append(p)
                    new_inbound[new_idx] = 1
                    idx_list.append(new_idx)
                    new_idx += 1
                new_dict[k] = idx_list

            if len(new_coords) == 0:
                self.coords = np.empty((0, 3))
            else:
                self.coords = np.array(new_coords, dtype=float)
            self.coords_dict = new_dict
            self.coords_inbound = new_inbound
            self.volume = volume_pad
        else:
            if coords.size > 0:
                inbound_idx = np.nonzero(
                    np.all(
                        (coords < np.array([z1, y1, x1])) * (coords >= np.array([z0, y0, x0])),
                        axis=1,
                    )
                )[0]
                coords_out = coords[inbound_idx] - np.array([z0, y0, x0])
                self.coords = coords_out

            self.volume = volume_pad

    def random_flip(self, axes):
        for flip_axis in axes:
            if random.randint(0, 1) == 1:
                self.volume = np.flip(self.volume, axis=flip_axis)
                if self.coords is not None and self.coords.ndim > 1:
                    self.coords[:, flip_axis] = (
                        self.volume.shape[flip_axis] - 1 - self.coords[:, flip_axis]
                    )

    def normalize_by_percentile(self, min_percentile, max_percentile):
        # check if torch.tensor, convert to numpy
        if isinstance(self.volume, torch.Tensor):
            self.volume = self.volume.numpy()
        for channel in range(self.volume.shape[0]):
            data = self.volume[channel]
            min = np.percentile(data, min_percentile)
            max = np.percentile(data, max_percentile)
            data = (data - min) / (max - min + 1e-6)
            self.volume[channel] = np.clip(data, 0, 1)

    def normalize_by_value(self, min_value, max_value):
        if isinstance(self.volume, torch.Tensor):
            self.volume = self.volume.numpy()
        for channel in range(self.volume.shape[0]):
            data = self.volume[channel]
            data = np.clip(data, min_value, max_value)
            data = (data - min_value) / (max_value - min_value + 1e-8)
            self.volume[channel] = data


class Transforms4Mask:
    def __init__(self, volume, mask, transforms, size=(128, 128, 128)):
        self.volume = volume
        self.mask = mask
        if isinstance(size, int):
            self.size = (size, size, size)
        else:
            self.size = size
        self.transforms = transforms
        self.params = {}

    def get_params(self, params):
        if "resize_ratio" in params.keys():
            self.params["resize_ratio"] = (
                np.random.uniform(*params["resize_ratio"]),
                np.random.uniform(*params["resize_ratio"]),
                np.random.uniform(*params["resize_ratio"]),
            )
            self.params["resize_mode"] = params.get("resize_mode", "trilinear")
        else:
            self.params["resize_ratio"] = (1, 1, 1)
            self.params["resize_mode"] = "trilinear"

        if "gamma_range" in params.keys():
            self.params["gamma"] = np.random.uniform(*params["gamma_range"])
        if "flip_axes" in params.keys():
            self.params["flip_axes"] = params["flip_axes"]

        if "rotate_degrees" in params.keys():
            self.params["rotate_degrees"] = np.random.uniform(*params["rotate_degrees"])
        else:
            self.params["rotate_degrees"] = np.random.uniform(0, 360)

        if "norm_percentiles" in params.keys():
            self.params["norm_percentiles"] = params["norm_percentiles"]
        else:
            self.params["norm_percentiles"] = (1, 99)

    def apply_transform(self):
        resize_flag = True if "resize" in self.transforms else False
        rotate_flag = True if "rotate" in self.transforms else False
        self.resize_and_rotate(
            resize_ratio=self.params["resize_ratio"],
            method=self.params["resize_mode"],
            rotate_degrees=self.params["rotate_degrees"],
            resize=resize_flag,
            rotate=rotate_flag,
        )
        if rotate_flag:
            self.center_crop(axial=False)

        if "gamma" in self.transforms:
            self.volume = self.volume ** self.params["gamma"]
        self.normalize_by_percentile(*self.params["norm_percentiles"])

        if "flip" in self.transforms:
            self.random_flip(self.params["flip_axes"])
        if "center_crop" in self.transforms:
            self.center_crop()
        elif "upperleft_crop" in self.transforms:
            self.upperleft_crop()
        elif "mode_crop" in self.transforms:
            self.mode_crop()
        elif "whole_volume" in self.transforms:
            return
        else:
            self.random_crop()

        self.mask = (self.mask > 0).astype(np.float32)

    def resize_and_rotate(self, resize_ratio, method, rotate_degrees, resize=False, rotate=False):
        target_size = tuple((self.volume.shape[-3:] * np.array(resize_ratio)).astype(int))

        volume = torch.from_numpy(self.volume)
        mask = torch.from_numpy(self.mask)
        ndim = volume.ndim

        for _ in range(5 - ndim):
            volume = volume.unsqueeze(0)

        if mask.ndim == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 4:
            mask = mask.unsqueeze(0)

        if resize:
            align_corners = None
            if method != "nearest":
                align_corners = True
            volume = F.interpolate(
                volume, size=target_size, align_corners=align_corners, mode=method
            )
            mask = F.interpolate(mask.float(), size=target_size, align_corners=None, mode="nearest")

        if rotate:
            for i in range(volume.shape[-3]):
                volume[..., i, :, :] = T.functional.rotate(
                    volume[..., i, :, :], rotate_degrees, interpolation=T.InterpolationMode.BILINEAR
                )
                mask[..., i, :, :] = T.functional.rotate(
                    mask[..., i, :, :], rotate_degrees, interpolation=T.InterpolationMode.NEAREST
                )

        for _ in range(5 - ndim):
            volume = volume[0]
        mask = mask[0, 0]

        self.volume = volume.numpy()
        self.mask = mask.numpy()

    def random_crop(self):
        in_chans = self.volume.shape[0]
        volume = self.volume
        mask = self.mask
        zmax, ymax, xmax = volume.shape[-3:]

        z0 = random.randint(0, max(0, zmax - self.size[0]))
        y0 = random.randint(0, max(0, ymax - self.size[1]))
        x0 = random.randint(0, max(0, xmax - self.size[2]))

        z1 = min(z0 + self.size[0], zmax)
        y1 = min(y0 + self.size[1], ymax)
        x1 = min(x0 + self.size[2], xmax)

        volume_crop = volume[:, z0:z1, y0:y1, x0:x1]
        mask_crop = mask[z0:z1, y0:y1, x0:x1]

        volume_pad = np.zeros(tuple([in_chans, *self.size]))
        mask_pad = np.zeros(tuple(self.size))
        volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume_crop
        mask_pad[0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = mask_crop

        self.volume = volume_pad
        self.mask = mask_pad

    def center_crop(self, axial=True):
        in_chans = self.volume.shape[0]
        volume = self.volume
        mask = self.mask
        zmax, ymax, xmax = volume.shape[-3:]

        y0 = (ymax - self.size[1]) // 2
        x0 = (xmax - self.size[2]) // 2

        y1 = min(y0 + self.size[1], ymax)
        x1 = min(x0 + self.size[2], xmax)
        if axial:
            z0 = (zmax - self.size[0]) // 2
            z1 = min(z0 + self.size[0], zmax)
            volume_pad = np.zeros(tuple([in_chans, *self.size]))
            mask_pad = np.zeros(tuple(self.size))
        else:
            z0, z1 = 0, zmax
            volume_pad = np.zeros(tuple([in_chans, (z1 - z0), *self.size[1:]]))
            mask_pad = np.zeros(tuple([(z1 - z0), *self.size[1:]]))

        volume_crop = volume[:, z0:z1, y0:y1, x0:x1]
        mask_crop = mask[z0:z1, y0:y1, x0:x1]

        volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume_crop
        mask_pad[0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = mask_crop

        self.volume = volume_pad
        self.mask = mask_pad

    def upperleft_crop(self):
        in_chans = self.volume.shape[0]
        volume = self.volume
        mask = self.mask
        zmax, ymax, xmax = volume.shape[-3:]

        z0 = y0 = x0 = 0
        z1 = min(z0 + self.size[0], zmax)
        y1 = min(y0 + self.size[1], ymax)
        x1 = min(x0 + self.size[2], xmax)

        volume_pad = np.zeros(tuple([in_chans, *self.size]))
        mask_pad = np.zeros(tuple(self.size))

        volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = volume[:, z0:z1, y0:y1, x0:x1]
        mask_pad[0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = mask[z0:z1, y0:y1, x0:x1]

        self.volume = volume_pad
        self.mask = mask_pad

    def mode_crop(self):
        def _mode_start(axis_coords, crop, limit):
            if axis_coords.size == 0 or limit == 0:
                return 0
            clipped = np.clip(axis_coords.astype(int), 0, max(0, limit - 1))
            mode_center = int(np.bincount(clipped, minlength=limit).argmax())
            start = mode_center - crop // 2
            return min(max(start, 0), max(0, limit - crop))

        mask_coords = np.column_stack(np.nonzero(self.mask > 0))
        zmax, ymax, xmax = self.mask.shape

        z0 = _mode_start(
            mask_coords[:, 0] if mask_coords.size else np.array([]), self.size[0], zmax
        )
        y0 = _mode_start(
            mask_coords[:, 1] if mask_coords.size else np.array([]), self.size[1], ymax
        )
        x0 = _mode_start(
            mask_coords[:, 2] if mask_coords.size else np.array([]), self.size[2], xmax
        )

        z1 = min(z0 + self.size[0], zmax)
        y1 = min(y0 + self.size[1], ymax)
        x1 = min(x0 + self.size[2], xmax)

        in_chans = self.volume.shape[0]
        volume_pad = np.zeros(tuple([in_chans, *self.size]))
        mask_pad = np.zeros(tuple(self.size))

        volume_pad[:, 0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = self.volume[
            :, z0:z1, y0:y1, x0:x1
        ]
        mask_pad[0 : (z1 - z0), 0 : (y1 - y0), 0 : (x1 - x0)] = self.mask[z0:z1, y0:y1, x0:x1]

        self.volume = volume_pad
        self.mask = mask_pad

    def random_flip(self, axes):
        for flip_axis in axes:
            if random.randint(0, 1) == 1:
                self.volume = np.flip(self.volume, axis=flip_axis)
                self.mask = np.flip(self.mask, axis=flip_axis)

    def normalize_by_percentile(self, min_percentile, max_percentile):
        if isinstance(self.volume, torch.Tensor):
            self.volume = self.volume.numpy()
        for channel in range(self.volume.shape[0]):
            data = self.volume[channel]
            min = np.percentile(data, min_percentile)
            max = np.percentile(data, max_percentile)
            data = (data - min) / (max - min + 1e-6)
            self.volume[channel] = np.clip(data, 0, 1)
