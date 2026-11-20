#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Explore axon-encoder pretraining on the copied single-channel EGFP images.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_pretrain_multidata.py \
--model mae_vit_base_patch16_1c_32 --vol_size 32 \
--data_path /media/hddshare/general/mae_brain/data/bmae/20240418_11_27_21_Xinyi_Thy1_EGFP_TH_Syto \
--marker_folder Ex_488_Em_0_precomputed \
--batch_size 8 --accum_iter 2 --mask_ratio 0.75 \
--epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 \
--output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/pre-train/brainmae-vitb-axon
