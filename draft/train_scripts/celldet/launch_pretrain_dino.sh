#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_pretrain_DINO.py \
--model mae_vit_base_patch16 \
--data_path /media/hddshare/general/mae_brain/data/bmae/20240418_11_27_21_Xinyi_Thy1_EGFP_TH_Syto \
--nuclei_folder Ex_642_Em_2_precomputed --marker_folder Ex_488_Em_0_precomputed \
--batch_size 32 --accum_iter 8 --epochs 800 --warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 --weight_decay_end 0.05 \
--global_crop_size 96 --local_crop_size 64 --local_crops_number 8 \
--num_of_data 10000 --transforms rfg \
--out_dim 65536 --student_temp 0.1 --teacher_temp 0.04 --momentum_teacher 0.996 \
--output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/pretrainDINO/pretrain_Syto_EGFP_BrainDINO_patch16_10kx800_0928
