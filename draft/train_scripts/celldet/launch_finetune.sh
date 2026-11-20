#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# reprod
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_finetune_hm_offset.py \
--batch_size 8 --accum_iter 2 \
--model mae_vit_base_patch16_4 \
--finetune /media/hddshare/general/mae_brain/brain-mae/output_dir/pre-train/brainmae-vitb/pretrain_Syto_EGFP_patch16_0602_checkpoint-799.pth \
--epochs 400 \
--blr 5e-4 \
--R 2 \
--focalloss_pos 0.85 \
--log_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/detection_0820/brainmae-vitb-pretrain_v3x512_EGFP_400_rfg_261109 \
--output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/detection_0820/brainmae-vitb-pretrain_v3x512_EGFP_400_rfg_261109 \
--data_path /media/hddshare/general/mae_brain/data/bmae \
--transforms rfg \
--wandb_project celldet_scaling_260820_dynamicdatasize \
--wandb_run_name brainmae-vitb-pretrain_v3x512_EGFP_400_rfg_261109 \
--train_data data/EGFP_200_3x512.json
