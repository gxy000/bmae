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
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        # if not torch.jit.is_scripting():
        #     # Dynamo doesn't support context managers yet
        #     is_dynamo_compiling = check_if_dynamo_compiling()
        #     if not is_dynamo_compiling:
        #         with warnings.catch_warnings(record=True):
        #             if x.numel() == 0 and self.training:
        #                 # https://github.com/pytorch/pytorch/issues/12013
        #                 assert not isinstance(
        #                     self.norm, torch.nn.SyncBatchNorm
        #                 ), "SyncBatchNorm does not support empty inputs!"

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

class UpsampleModule(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(input_dims, output_dims, kernel_size=2, stride=2)
        self.conv1 = Conv3d(output_dims, output_dims, kernel_size=1, bias=False, norm=LayerNorm(output_dims))
        self.conv2 = Conv3d(output_dims, output_dims, kernel_size=3, padding=1, bias=False, norm=LayerNorm(output_dims))

        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


MODEL_ARCH = '0710'

class CenterNet(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, dhw, emb_dim, decoder_embed_dim=512, patch_size=16, output_patch_size = 2):
        super(CenterNet, self).__init__()

        # self.d, self.h, self.w = dhw[0]*(patch_size//output_patch_size), dhw[1]*(patch_size//output_patch_size), dhw[2]*(patch_size//output_patch_size)
        self.d, self.h, self.w = dhw
        self.p = patch_size//output_patch_size
        # print(f'Initialize the mode, with p: {self.p}')
        self.hm_chans = 1
        self.off_chans = 3

        self.decoder_embed = nn.Linear(emb_dim, decoder_embed_dim, bias=True)
        # self.decoder_blocks_cg = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        # self.decoder_off = nn.Linear(decoder_embed_dim, self.p ** 3 * self.off_chans, bias=False)

        if MODEL_ARCH == '0708':
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
            self.decoder_hm = nn.Linear(decoder_embed_dim, self.p ** 3 * self.hm_chans, bias=False)  # decoder to patch
        elif MODEL_ARCH == '0710':
            upsample_modules = list()
            dim = decoder_embed_dim
            for i in range(int(math.log(self.p, 2))):
                output_dims = dim//2 if i < 3 else dim//4
                upsample_modules.append(UpsampleModule(dim, output_dims))
                dim = upsample_modules[-1].output_dims
            self.upsample_modules = nn.Sequential(*upsample_modules)
            self.decoder_hm = nn.Conv3d(dim, self.hm_chans, kernel_size=1, bias=False)  # decoder to patch
            self.decoder_off = nn.Conv3d(dim, self.off_chans, kernel_size=1, bias=False)  # decoder to patch

        else:
            raise ValueError(f'Unknown model architecture: {MODEL_ARCH}')

        # self.conv_shared = nn.Conv3d(decoder_embed_dim, 512, kernel_size=3, stride=self.anchor_stride, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.decoder_pred = nn.Linear(512, patch_size ** 3 * unpatchify_dim, bias=True)  # decoder to patch
        # self.conv_hm3x3 = nn.Conv3d(512, self.p ** 3 * self.hm_chans, kernel_size=3, stride=1, padding=1)
        # self.conv_hm1x1 = nn.Conv3d(self.p ** 3 * self.hm_chans, self.p ** 3 * 1, kernel_size=1, stride=1, padding=0)
        # self.conv_offset3 = nn.Conv3d(self.unpatchify_dim, self.unpatchify_dim, kernel_size=3, stride=1, padding=1)
        # self.conv_offset1 = nn.Conv3d(self.unpatchify_dim, 3, kernel_size=1, stride=1, padding=0)

        # self.conv_shared = nn.Conv3d(emb_dim, 512, kernel_size=3, stride=self.anchor_stride, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv_gflow = nn.Conv3d(512, 3, kernel_size=1, stride=1, padding=0)

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
        # x = self.conv_shared(x)
        # x = self.relu(x)

        # predictor projection
        # x = self.decoder_pred(x)

        if MODEL_ARCH == '0708':
            x = self.decoder_norm(x)
            x = x[:, 1: :]  # remove cls token
            pred_hm = self.decoder_hm(x)
            pred_hm = self.unpatchify(pred_hm, chans=self.hm_chans)
        elif MODEL_ARCH == '0710':
            x = x[:, 1:, :]  # remove cls token
            x = self.unpatchify_batch(x)
            x = self.upsample_modules(x)
            pred_hm = self.decoder_hm(x)
            pred_hm = pred_hm.reshape((pred_hm.shape[0], self.hm_chans, self.d*self.p, self.h*self.p, self.w*self.p))
            pred_off = self.decoder_off(x)
            pred_off = pred_off.reshape((pred_off.shape[0], self.off_chans, self.d*self.p, self.h*self.p, self.w*self.p))
        else:
            raise ValueError(f'Unknown model architecture: {MODEL_ARCH}')

        # pred_hm = self.relu(self.conv_hm3x3(x))
        # pred_hm = self.conv_hm1x1(pred_hm)
        # pred_hm = self.decoder_hm(x)
        # pred_hm = self.unpatchify(pred_hm, chans=self.hm_chans)
        # pred_off = self.decoder_off(x)
        # pred_off = self.unpatchify(pred_off, chans=self.off_chans)

        # pred_offset = self.relu(self.conv_offset3(x))
        # pred_offset = self.conv_offset1(pred_offset)


        return pred_hm, pred_off


class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, vox_size=128, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        vox_size = (vox_size, vox_size, vox_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (vox_size[0] // patch_size[0]) * (vox_size[1] // patch_size[1]) * (vox_size[2] // patch_size[2])
        self.vox_size = vox_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, Z, Y, X = x.shape
        # FIXME look at relaxing size constraints
        assert Z == self.vox_size[0] and Y == self.vox_size[1] and Z == self.vox_size[2], \
            f"Input image size ({Z}*{Y}*{X}) doesn't match model ({self.vox_size[0]}*{self.vox_size[1]}*{self.vox_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, vox_size=128, patch_size=16, in_chans=2,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 output_patch_size = 2,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.in_chans = in_chans
        self.p = output_patch_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.d = vox_size //patch_size
        self.h = vox_size // patch_size
        self.w = vox_size // patch_size
        self.patch_embed = PatchEmbed3D(vox_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # RPN specifics
        self.CenterNet = CenterNet((self.d, self.h, self.w), emb_dim=embed_dim, decoder_embed_dim = decoder_embed_dim, patch_size=patch_size, output_patch_size = self.p)
        self.relu = nn.ReLU(inplace=True)
        # self.conv_keypoint3 = nn.Conv3d(self.unpatchify_dim, self.unpatchify_dim, kernel_size = 3, stride=1, padding=1)
        # self.conv_keypoint1 = nn.Conv3d(self.unpatchify_dim, 3, kernel_size = 1, stride=1, padding=0)
        # self.conv_offset3 = nn.Conv3d(self.unpatchify_dim, self.unpatchify_dim, kernel_size=3, stride=1, padding=1)
        # self.conv_offset1 = nn.Conv3d(self.unpatchify_dim, 3, kernel_size = 1, stride=1, padding=0)
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

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

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

    def forward(self, voxel):
        latent = self.forward_encoder(voxel)
        pred_hm, pred_off = self.CenterNet(latent)
        pred_hm = _sigmoid(pred_hm)
        # x = self.unpatchify(x, chans=self.unpatchify_dim)
        # pred_keypoint = self.relu(self.conv_keypoint3(x))
        # pred_keypoint = self.conv_keypoint1(pred_keypoint)
        # pred_offset = self.relu(self.conv_offset3(x))
        # pred_offset = self.conv_offset1(pred_offset)

        return pred_hm, pred_off


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

def mae_vit_base_patch16_dec512d8b_4_1c(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, in_chans=1,
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
mae_vit_base_patch16_4_1c = mae_vit_base_patch16_dec512d8b_4_1c




mae_vit_base_patch16_8 = mae_vit_base_patch16_dec512d8b_8
mae_vit_base_patch16_16 = mae_vit_base_patch16_dec512d8b_16
#
#
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
