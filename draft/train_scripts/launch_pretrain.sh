#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# MAE
torchrun --standalone --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_pretrain.py \
--batch_size 32 --accum_iter 8 \
--model mae_vit_base_patch16 \
--mask_ratio 0.75 \
--epochs 800 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/pretrainDINO/pretrain_Syto_EGFP_BrainMAE_patch16_100kx800_1108 \
--data_path /media/hddshare/general/mae_brain/data/bmae/20240418_11_27_21_Xinyi_Thy1_EGFP_TH_Syto \
--nuclei_folder Ex_642_Em_2_precomputed \
--marker_folder Ex_488_Em_0_precomputed \
--wandb_project brainDINO-pretrain \
--wandb_run_name pretrain_Syto_EGFP_BrainMAE_patch16_100kx800_1108
