# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

import math
import torch.nn.functional as F
from util.bmae_utils import _sigmoid

def pred_to_boxes(output, hm_threshold=0.1, box_width=2):
    pred_hm = output["pred_hm"]
    pred_inst = output["pred_inst"]

    if pred_hm.dim() == 5 and pred_hm.size(1) == 1:
        pred_hm = pred_hm[:, 0]

    mask = pred_hm > hm_threshold
    if mask.sum() == 0:
        device = pred_hm.device
        embed_dim = pred_inst.shape[1]
        return (
            torch.empty((0, 6), device=device),
            torch.empty((0, 2), device=device),
            torch.empty((0, embed_dim), device=device),
        )

    bzyx = torch.nonzero(mask, as_tuple=False)
    _, z_idx, y_idx, x_idx = bzyx[:, 0], bzyx[:, 1], bzyx[:, 2], bzyx[:, 3]
    d, h, w = pred_hm.shape[-3:]
    norm = torch.tensor([d, h, w], device=pred_hm.device)
    z = (z_idx+0.5).float() / norm[0]
    y = (y_idx+0.5).float() / norm[1]
    x = (x_idx+0.5).float() / norm[2]
    size = float(box_width)
    sizes = (torch.tensor([size, size, size], device=pred_hm.device) / norm).expand(z.numel(), 3)
    boxes = torch.stack((z, y, x), dim=1)
    boxes = torch.cat((boxes, sizes), dim=1)

    pred_inst = pred_inst.permute(0, 2, 3, 4, 1)
    embeds = pred_inst[mask]

    scores = pred_hm[mask].clamp(min=1e-6, max=1 - 1e-6)
    cls_logits = torch.stack((scores, 1.0 - scores), dim=-1).log()
    return boxes, cls_logits, embeds

def assign_instance_ids(pred_hm, pred_inst, hm_threshold=0.1, radius=0.5):
    """
    Assign instance ids on the heatmap mask, returning an instance map with the same shape as pred_hm.
    pred_hm: [B, 1, D, H, W] or [B, D, H, W]
    pred_inst: [B, C, D, H, W] embeddings
    """
    if pred_hm.dim() == 5 and pred_hm.size(1) == 1:
        pred_hm = pred_hm[:, 0]

    mask = pred_hm > hm_threshold
    if mask.sum() == 0:
        return torch.full_like(pred_hm, -1, dtype=torch.long)

    pred_inst = pred_inst.permute(0, 2, 3, 4, 1)
    embeds = pred_inst[mask]

    # greedy radius clustering on embeddings
    instance_ids = -torch.ones(len(embeds), dtype=torch.long, device=pred_hm.device)
    centroids = []
    for i, emb in enumerate(embeds):
        placed = False
        for cid, c in enumerate(centroids):
            dist = torch.dist(emb, c)
            if dist < radius:
                instance_ids[i] = cid
                centroids[cid] = (c + emb) / 2
                placed = True
                break
        if not placed:
            centroids.append(emb.clone())
            instance_ids[i] = len(centroids) - 1

    instance_map = torch.full_like(pred_hm, -1, dtype=torch.long)
    instance_map[mask] = instance_ids
    return instance_map


def _neg_loss(pred, gt, threshold_pos, neg_scale=1): #, alpha=0.5):
    pos_inds = gt.ge(threshold_pos).float()
    neg_inds = gt.lt(threshold_pos).float()
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum() * neg_scale

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / (num_pos + 1e-6)

def FocalLoss(out, target, threshold_pos=1):
    return _neg_loss(out, target, threshold_pos) #, alpha=0.9)

class SetCriterion(nn.Module):
    """
    Criterion for dense axon segmentation + instance embeddings.

    outputs:
        pred_hm   : [B, 1, D, H, W] (sigmoid applied)
        pred_inst : [B, Cemb, D, H, W]

    targets[b]:
        heatmap      : [D, H, W] float in [0,1]
        instance_map : [D, H, W] int64, background = -1
    """

    def __init__(
        self,
        weight_dict=None,
        threshold_pos=1.0,
        delta_var=0.5,
        delta_dist=1.5,
        alpha=1.0,
        beta=1.0,
        gamma=0.001,
        max_samples=4096,
        max_per_inst=512,
        min_vox_per_inst=10,
        weighted_disc_loss=False,
    ):
        super().__init__()
        self.weight_dict = weight_dict or {
            "loss_hm": 1.0,
            "loss_de": 1.0,
        }

        self.threshold_pos = threshold_pos

        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.de_alpha = alpha
        self.de_beta = beta
        self.de_gamma = gamma

        self.max_samples = max_samples
        self.max_per_inst = max_per_inst
        self.min_vox_per_inst = min_vox_per_inst
        self.weighted_disc_loss = weighted_disc_loss
        self.p0 = None

    def _get_p0(self, embed_dim, device, dtype):
        if self.p0 is None or self.p0.numel() != embed_dim or self.p0.device != device or self.p0.dtype != dtype:
            self.p0 = nn.Parameter(torch.zeros(embed_dim, device=device, dtype=dtype))
        return self.p0

    def _disc_embedding_loss(
        self,
        embeddings,
        instance_ids,
        coords=None,
        delta_var=0.5,
        delta_dist=1.5,
        alpha=1.0,
        beta=1.0,
        gamma=0.001,
        junction_id=0,
        sigma=3.0,
        junction_weight=3.0,
        w_max=10.0,
        delta_j=0.0,
        margin0=1.0,
        lam_junc=1.0,
        lam_sep0=1.0,
    ):
        device = embeddings.device
        if not self.weighted_disc_loss:
            valid = instance_ids >= 0
            if valid.sum() == 0:
                return torch.tensor(0.0, device=device)

            embeds = embeddings[valid]
            ids = instance_ids[valid]
            uniq = ids.unique()
            n_inst = len(uniq)

            means = torch.stack([embeds[ids == u].mean(0) for u in uniq])

            var_terms = []
            for u, mean in zip(uniq, means):
                d = torch.norm(embeds[ids == u] - mean, dim=1)
                var_terms.append(torch.mean(torch.clamp(d - delta_var, min=0.0) ** 2))
            loss_var = torch.stack(var_terms).mean()

            if n_inst > 1:
                dist = torch.cdist(means, means)
                mask = ~torch.eye(n_inst, device=device).bool()
                loss_dist = torch.mean(torch.clamp(2 * delta_dist - dist[mask], min=0.0) ** 2)
            else:
                loss_dist = torch.tensor(0.0, device=device)

            loss_reg = torch.norm(means, dim=1).mean()

            return alpha * loss_var + beta * loss_dist + gamma * loss_reg

        valid = instance_ids >= 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        if coords is None:
            raise ValueError("coords is required when weighted=True")

        embeds = embeddings[valid]
        ids = instance_ids[valid]
        coords = coords[valid].to(device=device, dtype=torch.float32)

        j_mask = ids == junction_id
        inst_mask = ids > junction_id

        p0 = self._get_p0(embeds.shape[1], device, embeds.dtype)

        disc_loss = torch.tensor(0.0, device=device)
        means = torch.empty((0, embeds.shape[1]), device=device, dtype=embeds.dtype)
        n_inst = 0
        if inst_mask.sum() > 0:
            embeds_i = embeds[inst_mask]
            ids_i = ids[inst_mask]
            coords_i = coords[inst_mask]

            if j_mask.sum() > 0:
                coords_j = coords[j_mask]
                sigma_val = max(float(sigma), 1e-6)
                dmat = torch.cdist(coords_i, coords_j)
                dmin = dmat.min(dim=1).values
                w = 1.0 + junction_weight * torch.exp(-(dmin / sigma_val) ** 2)
                if w_max is not None:
                    w = w.clamp(max=w_max)
            else:
                w = torch.ones((embeds_i.shape[0],), device=device, dtype=torch.float32)

            uniq = ids_i.unique()
            n_inst = len(uniq)

            means_list = []
            for u in uniq:
                m = ids_i == u
                wu = w[m].unsqueeze(1)
                eu = embeds_i[m]
                denom = wu.sum(dim=0).clamp_min(1e-6)
                mu = (wu * eu).sum(dim=0) / denom
                means_list.append(mu)
            means = torch.stack(means_list, dim=0)

            var_terms = []
            for k, u in enumerate(uniq):
                m = ids_i == u
                eu = embeds_i[m]
                wu = w[m]
                d = torch.norm(eu - means[k], dim=1)
                per = torch.clamp(d - delta_var, min=0.0) ** 2
                var_u = (wu * per).sum() / wu.sum().clamp_min(1e-6)
                var_terms.append(var_u)
            loss_var = torch.stack(var_terms).mean()

            if n_inst > 1:
                dist = torch.cdist(means, means)
                mask = ~torch.eye(n_inst, device=device).bool()
                loss_dist = torch.mean(torch.clamp(2 * delta_dist - dist[mask], min=0.0) ** 2)
            else:
                loss_dist = torch.tensor(0.0, device=device)

            loss_reg = torch.norm(means, dim=1).mean()
            disc_loss = alpha * loss_var + beta * loss_dist + gamma * loss_reg

        if j_mask.any():
            e0 = embeds[j_mask]
            d0 = torch.norm(e0 - p0[None, :], dim=1)
            loss_junc = torch.mean(torch.clamp(d0 - delta_j, min=0.0) ** 2)
        else:
            loss_junc = torch.tensor(0.0, device=device)

        if n_inst > 0:
            dmu = torch.norm(means - p0[None, :], dim=1)
            loss_sep0 = torch.mean(torch.clamp(margin0 - dmu, min=0.0) ** 2)
        else:
            loss_sep0 = torch.tensor(0.0, device=device)

        return disc_loss + lam_junc * loss_junc + lam_sep0 * loss_sep0

    def _sample_voxels(self, emb, inst):
        device = emb.device
        valid = inst >= 0
        if valid.sum() == 0:
            return None, None, None

        ids_all = inst[valid]
        emb_all = emb.permute(1, 2, 3, 0)[valid]
        coords_all = torch.nonzero(valid, as_tuple=False)

        uniq = ids_all.unique()
        keep = []
        for u in uniq:
            if (ids_all == u).sum() >= self.min_vox_per_inst:
                keep.append(u)
        if len(keep) == 0:
            return None, None, None

        sel = []
        for u in keep:
            idx = torch.nonzero(ids_all == u, as_tuple=False).squeeze(1)
            k = min(len(idx), self.max_per_inst)
            if k < len(idx):
                idx = idx[torch.randperm(len(idx), device=device)[:k]]
            sel.append(idx)

        sel = torch.cat(sel)
        if sel.numel() > self.max_samples:
            sel = sel[torch.randperm(sel.numel(), device=device)[:self.max_samples]]

        return emb_all[sel], ids_all[sel], coords_all[sel]

    def forward(self, outputs, targets):
        pred_hm = outputs["pred_hm"]
        pred_inst = outputs["pred_inst"]

        if pred_hm.dim() == 5 and pred_hm.size(1) == 1:
            pred_hm = pred_hm[:, 0]

        device = pred_inst.device
        B = pred_inst.size(0)

        gt_hm = torch.stack([t["heatmap"] for t in targets]).to(device)
        gt_inst = torch.stack([t["instance_map"] for t in targets]).to(device)

        loss_hm = FocalLoss(pred_hm, gt_hm, self.threshold_pos) # set threshold_pos to a smaller value

        loss_de_sum = torch.tensor(0.0, device=device)
        used = 0
        for b in range(B):
            emb_b = pred_inst[b]
            inst_b = gt_inst[b]

            emb_s, ids_s, coords_s = self._sample_voxels(emb_b, inst_b)
            if emb_s is None:
                continue

            ids_s = ids_s.clone()
            ids_s[ids_s > 0] += b * 100000
            loss_de_sum += self._disc_embedding_loss(
                emb_s,
                ids_s,
                coords=coords_s,
                delta_var=self.delta_var,
                delta_dist=self.delta_dist,
                alpha=self.de_alpha,
                beta=self.de_beta,
                gamma=self.de_gamma,
            )
            used += 1

        loss_de = loss_de_sum / max(used, 1)

        return {
            "loss_hm": loss_hm * self.weight_dict.get("loss_hm", 1.0),
            "loss_de": loss_de * self.weight_dict.get("loss_de", 1.0),
        }



class Conv3d(torch.nn.Conv3d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

UPSAMPLE_ARCH = '0203'

class UpsampleModule(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        if UPSAMPLE_ARCH == '0204':
            mode = 'trilinear'
            # print(f'Upsample by interpolate')
            self.upsample = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            self.conv1 = Conv3d(input_dims, output_dims, kernel_size=1, bias=False, norm=LayerNorm(output_dims))
        else:
            self.conv_transpose = nn.ConvTranspose3d(input_dims, output_dims, kernel_size=2, stride=2)
            self.conv1 = Conv3d(output_dims, output_dims, kernel_size=1, bias=False, norm=LayerNorm(output_dims))
        self.conv2 = Conv3d(output_dims, output_dims, kernel_size=3, padding=1, bias=False, norm=LayerNorm(output_dims))

        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        # x = self.conv_transpose(x)
        x = self.upsample(x) if UPSAMPLE_ARCH == '0204' else self.conv_transpose(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class CenterNet(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, dhw, emb_dim, decoder_embed_dim=512, instance_dim=16, patch_size=16, output_patch_size=2):
        super(CenterNet, self).__init__()

        # self.d, self.h, self.w = dhw[0]*(patch_size//output_patch_size), dhw[1]*(patch_size//output_patch_size), dhw[2]*(patch_size//output_patch_size)
        self.d, self.h, self.w = dhw
        self.p = patch_size//output_patch_size
        self.hm_chans = 1
        self.inst_chans = instance_dim

        self.decoder_embed = nn.Linear(emb_dim, decoder_embed_dim, bias=True)


        upsample_modules = list()
        dim = decoder_embed_dim
        for i in range(int(math.log(self.p, 2))):
            output_dims = dim//2 if i < 3 else dim//4
            upsample_modules.append(UpsampleModule(dim, output_dims))
            dim = upsample_modules[-1].output_dims
        self.upsample_modules = nn.Sequential(*upsample_modules)
        self.decoder_hm = nn.Conv3d(dim, self.hm_chans, kernel_size=1, bias=False)  # per voxel: patch=1
        self.decoder_inst = nn.Conv3d(dim, self.inst_chans, kernel_size=1, bias=False)  # per voxel: patch=1, dim=16

    def unpatchify(self, voxel, chans):
        """
        input voxel: (N, L, patch_size**3 * chans)
        output voxel: (N, chans, Z, H, W)

        """
        assert self.d * self.h * self.w == voxel.shape[1]
        voxel = voxel.reshape(shape=(voxel.shape[0], self.d, self.h, self.w, self.p, self.p, self.p, chans))
        voxel = torch.einsum('ndhwpqrc->ncdphqwr', voxel)
        voxel = voxel.reshape(shape=(voxel.shape[0], chans, self.d * self.p, self.h * self.p, self.w * self.p))

        return voxel

    def unpatchify_batch(self, voxel):
        """
        Input voxel: (N, L, channels)
        Output voxel: (N, chans, Z, H, W)
        """

        chans = voxel.shape[-1]
        voxel = voxel.reshape(shape=(voxel.shape[0], self.d, self.h, self.w, chans))
        voxel = torch.einsum('ndhwc->ncdhw', voxel)
        return voxel.contiguous()

    def forward(self, x):
        x = self.decoder_embed(x)
        x = x[:, 1:, :]  # remove cls token
        x = self.unpatchify_batch(x)
        x = self.upsample_modules(x)
        pred_hm = self.decoder_hm(x)
        pred_hm = pred_hm.reshape((pred_hm.shape[0], self.hm_chans, self.d*self.p, self.h*self.p, self.w*self.p))
        pred_inst = self.decoder_inst(x)
        pred_inst = pred_inst.reshape((pred_inst.shape[0], self.inst_chans, self.d*self.p, self.h*self.p, self.w*self.p))
    
        pred_hm = _sigmoid(pred_hm)
        return {"pred_hm": pred_hm, "pred_inst": pred_inst}

class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, vox_size=(128, 128, 128), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # vox_size = (vox_size, vox_size, vox_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (vox_size[0] // patch_size[0]) * (vox_size[1] // patch_size[1]) * (vox_size[2] // patch_size[2])
        self.vox_size = vox_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, Z, Y, X = x.shape
        # FIXME look at relaxing size constraints
        assert Z == self.vox_size[0] and Y == self.vox_size[1] and X == self.vox_size[2], \
            f"Input image size ({Z}*{Y}*{X}) doesn't match model ({self.vox_size[0]}*{self.vox_size[1]}*{self.vox_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, vox_size=(128, 128, 128), patch_size=16, in_chans=2,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 output_patch_size = 2,
                 instance_dim=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.in_chans = in_chans
        self.p = output_patch_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.d = vox_size[0] // patch_size
        self.h = vox_size[1] // patch_size
        self.w = vox_size[2] // patch_size
        self.patch_embed = PatchEmbed3D(vox_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.dataset_token = torch.ones(1, num_patches + 1, embed_dim)

        self.CenterNet = CenterNet((self.d, self.h, self.w), emb_dim=embed_dim, decoder_embed_dim = decoder_embed_dim, patch_size=patch_size, output_patch_size = self.p, instance_dim=instance_dim)
        self.relu = nn.ReLU(inplace=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, dataset_idx):
        # embed patches
        x = self.patch_embed(x)
        embed = self.pos_embed + self.dataset_token.to(x.device)*dataset_idx[:, None, None]

        # add pos embed w/o cls token
        x = x + embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, center_gradient, pred_flow):
        target = self.patchify(center_gradient, chans=3)
        criterion = nn.MSELoss(reduction="mean")
        return criterion(target, pred_flow)

    def forward(self, voxel, dataset_idx):
        latent = self.forward_encoder(voxel, dataset_idx)
        return self.CenterNet(latent)

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_4(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_o1(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, output_patch_size=1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_4_i32(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        instance_dim=32,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b_1c(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, in_chans=1,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_1c_32(**kwargs):
    model = MaskedAutoencoderViT(
        vox_size=(32, 256, 256),
        patch_size=16, embed_dim=768, depth=12, num_heads=12, in_chans=1, output_patch_size=2,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_1c_32_o1(**kwargs):
    model = MaskedAutoencoderViT(
        vox_size=(32, 256, 256),
        patch_size=16, embed_dim=768, depth=12, num_heads=12, in_chans=1, output_patch_size=1,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch32_dec512d8b_1c_32(**kwargs):
    model = MaskedAutoencoderViT(
        vox_size=(32, 256, 256),
        patch_size=32, embed_dim=768, depth=12, num_heads=12, in_chans=1,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_8(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b_16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16_4 = mae_vit_base_patch16_dec512d8b_4
mae_vit_base_patch16_o1 = mae_vit_base_patch16_dec512d8b_o1
mae_vit_base_patch16_4_i32 = mae_vit_base_patch16_dec512d8b_4_i32
mae_vit_base_patch16_1c = mae_vit_base_patch16_dec512d8b_1c
mae_vit_base_patch16_1c_32 = mae_vit_base_patch16_dec512d8b_1c_32
mae_vit_base_patch16_1c_32_o1 = mae_vit_base_patch16_dec512d8b_1c_32_o1
mae_vit_base_patch32_1c_32 = mae_vit_base_patch32_dec512d8b_1c_32



mae_vit_base_patch16_8 = mae_vit_base_patch16_dec512d8b_8
mae_vit_base_patch16_16 = mae_vit_base_patch16_dec512d8b_16
#
#
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
