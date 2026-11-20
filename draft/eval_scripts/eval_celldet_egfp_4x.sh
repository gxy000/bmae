#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python test_evaluate_cn_wholev.py \
--data_path /media/hddshare/general/mae_brain/data/bmae \
--testdata EGFP_200_eval4_0915 \
--ckpt_folders detection_0820 detetion \
--model_archs mae_vit_base_patch16_4 mae_vit_base_patch16_4 \
--models DINO10k_v3x512_EGFP_400_rfg_261011 \
pretrain_8_512_cn_hm_offset_g8_16_2_1d_85fl_rfg_0819_v3x512_EGFP_400 \
--checkpoints checkpoint-399.pth checkpoint-399.pth \
--plot_labels DINO10k_v3_400 BMAE100k_v3_400 \
--threshold_hm 0.3 \
--plot_R 10 \
--R 0 16 \
--dist 2
