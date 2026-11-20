# Copyright (c) Meta Platforms, Inc. and affiliates.
# DINO-specific code adapted from facebookresearch/dino.

from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


def _trunc_normal_(tensor, mean=0.0, std=1.0):
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std)
    with torch.no_grad():
        return tensor.normal_(mean, std)


class PatchEmbed3D(nn.Module):
    """BrainMAE-compatible 3D patch embedding with variable-size forward support."""

    def __init__(self, vox_size=128, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(vox_size, int):
            vox_size = (vox_size, vox_size, vox_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (
            (vox_size[0] // patch_size[0])
            * (vox_size[1] // patch_size[1])
            * (vox_size[2] // patch_size[2])
        )
        self.vox_size = vox_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, z, y, xdim = x.shape
        pz, py, px = self.patch_size
        assert z % pz == 0 and y % py == 0 and xdim % px == 0, (
            f"Input size ({z}*{y}*{xdim}) must be divisible by patch size " f"({pz}*{py}*{px})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class BrainDINOEncoder(nn.Module):
    """ViT encoder with the same parameter names/shapes as the BrainMAE encoder.

    The stored positional embedding remains the BrainMAE 128^3 / 8x8x8 table.
    During DINO pretraining it is interpolated in 3D for 96^3 and 64^3 crops.
    """

    def __init__(
        self,
        vox_size=128,
        patch_size=16,
        in_chans=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(vox_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _cube_grid_size(num_tokens):
        side = round(num_tokens ** (1.0 / 3.0))
        if side ** 3 != num_tokens:
            raise ValueError(f"Expected a cubic 3D positional grid, got {num_tokens} tokens.")
        return side

    def interpolate_pos_encoding(self, grid_size):
        patch_pos_embed = self.pos_embed[:, 1:, :]
        num_base_patches = patch_pos_embed.shape[1]
        base_side = self._cube_grid_size(num_base_patches)
        if tuple(grid_size) == (base_side, base_side, base_side):
            return self.pos_embed

        dim = patch_pos_embed.shape[-1]
        pos_tokens = patch_pos_embed.reshape(1, base_side, base_side, base_side, dim)
        pos_tokens = pos_tokens.permute(0, 4, 1, 2, 3)
        pos_tokens = F.interpolate(
            pos_tokens, size=tuple(grid_size), mode="trilinear", align_corners=False,
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).reshape(1, -1, dim)
        return torch.cat((self.pos_embed[:, :1, :], pos_tokens), dim=1)

    def forward_tokens(self, x):
        _, _, z, y, xdim = x.shape
        pz, py, px = self.patch_embed.patch_size
        grid_size = (z // pz, y // py, xdim // px)
        x = self.patch_embed(x)

        pos_embed = self.interpolate_pos_encoding(grid_size)
        x = x + pos_embed[:, 1:, :]

        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        return self.forward_tokens(x)[:, 0]


class DINOHead(nn.Module):
    """Original DINO projection head."""

    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class BrainDINO(nn.Module):
    def __init__(
        self,
        encoder,
        out_dim=65536,
        use_bn_in_head=False,
        norm_last_layer=True,
        head_nlayers=3,
        head_hidden_dim=2048,
        head_bottleneck_dim=256,
    ):
        super().__init__()
        self.backbone = encoder
        self.head = DINOHead(
            encoder.embed_dim,
            out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
            nlayers=head_nlayers,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
        )

    def forward_single(self, x):
        return self.head(self.backbone(x))

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            return self.forward_single(x)

        # Process consecutive crops with the same 3D shape together, preserving crop order.
        output = []
        start = 0
        while start < len(x):
            crop_size = tuple(x[start].shape[-3:])
            end = start + 1
            while end < len(x) and tuple(x[end].shape[-3:]) == crop_size:
                end += 1
            output.append(self.forward_single(torch.cat(x[start:end], dim=0)))
            start = end
        return torch.cat(output, dim=0)

    def encoder_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()}


class DINOLoss(nn.Module):
    """Original DINO cross-view self-distillation loss with centering."""

    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            world_size = dist.get_world_size()
        batch_center = batch_center / (len(teacher_output) * world_size)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def _make_dino(
    vox_size=128,
    patch_size=16,
    in_chans=2,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    **kwargs,
):
    encoder = BrainDINOEncoder(
        vox_size=vox_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        norm_layer=norm_layer,
    )
    return BrainDINO(encoder, **kwargs)


def dino_vit_base_patch16(**kwargs):
    return _make_dino(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)


def dino_vit_base_patch16_1c(**kwargs):
    return _make_dino(patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, **kwargs)


def dino_vit_small_patch16(**kwargs):
    return _make_dino(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)


def dino_vit_small_patch16_1c(**kwargs):
    return _make_dino(patch_size=16, in_chans=1, embed_dim=384, depth=12, num_heads=6, **kwargs)


def dino_vit_tiny_patch16(**kwargs):
    return _make_dino(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)


def dino_vit_tiny_patch16_1c(**kwargs):
    return _make_dino(patch_size=16, in_chans=1, embed_dim=192, depth=12, num_heads=3, **kwargs)


# MAE-name aliases let the new DINO main script reuse existing launch-script model names.
mae_vit_base_patch16 = dino_vit_base_patch16
mae_vit_base_patch16_1c = dino_vit_base_patch16_1c
mae_vit_small_patch16 = dino_vit_small_patch16
mae_vit_small_patch16_1c = dino_vit_small_patch16_1c
mae_vit_tiny_patch16 = dino_vit_tiny_patch16
mae_vit_tiny_patch16_1c = dino_vit_tiny_patch16_1c
