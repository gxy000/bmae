import numpy as np
import torch
import cv2
import imageio
from matplotlib.cm import get_cmap
from pathlib import Path


def get_rgb_color(color):
    """
    Convert a color name to an RGB tuple.
    Args:
        color (str): Color name.
    Returns:
        tuple: RGB color tuple.
    """
    if color == "w":
        return (255, 255, 255)
    elif color == "g":
        return (0, 255, 0)
    elif color == "c":
        return (0, 255, 255)
    elif color == "r":
        return (255, 0, 0)
    elif color == "b":
        return (0, 0, 255)
    else:
        raise ValueError(f"Unknown color: {color}")


class Visualizer:
    def __init__(self, channels, captions=None):
        """
        :param channels: C * Z * Y * X
        :param captions: list of str
        """
        if isinstance(channels, torch.Tensor):
            channels = channels.cpu().numpy()
        self.channels = channels
        self.ind_rgbs = {}
        self.cmap = get_cmap("tab20")
        if captions is not None:
            for caption in captions:
                self.add_one_viz(caption)

    def channels_to_rgb(self, caption, new_channels=None, is_rgb=False):
        """
        transform 2 channel array to rgb 3D image,
        add caption at bottom and add white edge padding to the right
        Args:
            caption (str): Must be "marker_only" for single channel viz of cell
        """
        if new_channels is None:
            channels = self.channels
            assert not is_rgb, "Voxels are not RGB format"
        else:
            channels = new_channels

        if is_rgb:
            rgb = np.transpose(channels, (1, 2, 3, 0))
        else:
            nuclei = caption == "2 channels"
            zero_channel = np.zeros(channels.shape[1:])
            if nuclei:
                rgb = np.stack(arrays=[channels[1], zero_channel, channels[0] * 0.5], axis=-1)
            else:
                rgb = np.stack(arrays=[channels[-1], zero_channel, zero_channel], axis=-1)

        z, y, x = rgb.shape[:3]
        text_padding = np.ones((32, x, 3))

        # get boundary of this text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_width = 1
        textsize = cv2.getTextSize(caption, font, text_scale, text_width)[0]

        # get coords based on boundary
        textX = (text_padding.shape[1] - textsize[0]) // 2
        textY = (text_padding.shape[0] + textsize[1]) // 2

        # add text centered on image
        cv2.putText(text_padding, caption, (textX, textY), font, text_scale, (0, 0, 0), text_width)

        text_padding = np.repeat(text_padding[np.newaxis, ...], z, axis=0)
        edge_padding = np.ones((y + 32, 8, 3))
        edge_padding = np.repeat(edge_padding[np.newaxis, ...], z, axis=0)

        rgb = np.concatenate([rgb, text_padding], axis=1)  # pad text below rgb image
        rgb = np.concatenate([rgb, edge_padding], axis=2)

        return (rgb * 255).astype(np.uint8)

    def add_one_viz(self, caption, new_channels=None, is_rgb=False):
        self.ind_rgbs[caption] = self.channels_to_rgb(caption, new_channels, is_rgb=is_rgb)

    def draw_annotations(self, labels, caption, box_width=16, draw_box=True, color="w"):
        """
        Args:
            labels (ndarray): N * 3, start point of bounding box
            box_width (int): fixed size bbox width
            draw_box (bool, optional): if True, draw bbox, otherwise box center
        """
        if labels.shape[0] < 1:
            return
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels = labels[:, :3]
        # check if <caption> is in self.ind_rgbs.keys()
        if caption not in self.ind_rgbs.keys():
            self.add_one_viz(caption)
        out = self.ind_rgbs[caption]
        z, y, x = self.channels.shape[1:]

        color_rgb = get_rgb_color(color)

        if draw_box:
            # braw bbox, random color
            for box in labels:
                box = box.astype(int)
                for i in range(max(box[0] - box_width, 0), min(box[0] + box_width, z)):
                    # transpose (y,x) to adapt to cv2
                    top_left_corner = (box[2], box[1])
                    bottom_right_corner = (min(box[2] + box_width, x), min(box[1] + box_width, y))
                    cv2.rectangle(out[i], top_left_corner, bottom_right_corner, color_rgb, 1)
        else:
            # plot box center as white dot
            for box in labels:
                center = box.astype(int) + box_width // 2
                for i in range(
                    max(center[0] - box_width // 2, 0), min(center[0] + box_width // 2, z)
                ):
                    # transpose (y,x) to adapt to cv2
                    cv2.circle(
                        out[i], (center[2], center[1]), radius=1, color=color_rgb, thickness=-1
                    )

        self.ind_rgbs[caption] = out

    def draw_axon_bbox(self, tgt, caption):
        """
        Args:
            tgt : dict of data
            box_width (int): fixed size bbox width
            draw_box (bool, optional): if True, draw bbox, otherwise box center
        """
        if len(tgt.items()) < 1:
            return
        # check if <caption> is in self.ind_rgbs.keys()
        if caption not in self.ind_rgbs.keys():
            self.add_one_viz(caption)
        out = self.ind_rgbs[caption]
        z, y, x = self.channels.shape[1:]

        instance_colors = {
            inst_id: tuple((np.array(self.cmap(i % 20)[:3]) * 255).astype(int))
            for i, inst_id in enumerate(np.unique(tgt["instances"]))
        }
        boxes = tgt["boxes"].cpu().tolist()
        labels = tgt["labels"].cpu().tolist()
        instance_ids = tgt["instances"].cpu().tolist()

        for box, label, inst_id in zip(boxes, labels, instance_ids):
            if label == 0:
                continue
            cz, cy, cx, dz, dh, dw = box
            z0 = (cz - dz / 2) * z
            d = dz * z

            # convert normalized box to pixel coords
            y0 = int((cy - dh / 2) * y)
            x0 = int((cx - dw / 2) * x)
            h = int(dh * y)
            w = int(dw * x)
            top_left_corner = (x0, y0)
            bottom_right_corner = (min(x0 + w, x), min(y0 + h, y))

            color = tuple(map(int, instance_colors[inst_id]))
            for i, j in enumerate(range(max(int(z0), 0), min(int(z0 + d), z))):
                cv2.rectangle(out[j], top_left_corner, bottom_right_corner, color, 1)

        self.ind_rgbs[caption] = out

    def draw_axon_segments(self, tgt, caption):
        """
        Args:
            tgt : dict of segments {instance_id: list of (pt1, pt2)}
        """
        if len(tgt.items()) < 1:
            return
        # check if <caption> is in self.ind_rgbs.keys()
        if caption not in self.ind_rgbs.keys():
            self.add_one_viz(caption)
        out = self.ind_rgbs[caption]
        z, y, x = self.channels.shape[1:]

        instance_colors = {
            inst_id: tuple((np.array(self.cmap(i % 20)[:3]) * 255).astype(int))
            for i, inst_id in enumerate(tgt.keys())
        }

        for inst_id, centerline in tgt.items():
            if len(centerline) == 0:
                continue
            if len(centerline[0]) == 3:
                segments = []
                for i in range(len(centerline) - 1):
                    segments.append((centerline[i], centerline[i + 1]))
            else:
                segments = centerline
            for ((cz1, cy1, cx1), (cz2, cy2, cx2)) in segments:
                p1 = (min(int(cx1 * x), x), min(int(cy1 * y), y))
                p2 = (min(int(cx2 * x), x), min(int(cy2 * y), y))

                color = tuple(map(int, instance_colors[inst_id]))
                z_start = max(min(int(cz1 * z), int(cz2 * z)), 0)
                z_end = min(max(int(cz1 * z), int(cz2 * z)), z)
                for j in range(z_start, z_end):
                    cv2.line(out[j], p1, p2, color, thickness=5, lineType=cv2.LINE_AA)

        self.ind_rgbs[caption] = out

    def write_gif(self, file_name):
        out = np.concatenate([np.array(v) for v in self.ind_rgbs.values()], axis=-2)
        imageio.mimsave(file_name, [xy for xy in out], duration=0.3)

    def write_max_projection_png(self, out_dir, step_size=8):
        """
        write max projection of each channel to png
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out = np.concatenate([np.array(v) for v in self.ind_rgbs.values()], axis=-2)
        # max projection every step_size
        for i in range(0, out.shape[0], step_size):
            if i + step_size > out.shape[0]:
                break
            max_proj = np.max(out[i : i + step_size], axis=0)
            imageio.imwrite(f"{out_dir}/max_proj_{i}.png", max_proj)
