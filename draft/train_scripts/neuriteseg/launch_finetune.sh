#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Run from the brain-mae repository root; paths use the shared lab copy.
# point2seg, 2e-4, all loss function parameters set default
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_finetune_axon_hm_multidata.py \
--batch_size 8 --accum_iter 2 \
--model mae_vit_base_patch16_1c_32 \
--finetune /media/hddshare/general/mae_brain/brain-mae/output_dir/pre-train/neun_gfap_patch16_w_dataset_token_ALL0_32x256_251030_h100s_checkpoint-799.pth \
--epochs 400 \
--warmup_epochs 50 \
--blr 2e-4 \
--log_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/axon_hm/brainmae-vitb-pretrain_32x256_patch16_w_dataset_token_ALL0_instanceMULTI_overlayvol1_rfg4096_Segments_SegNetData0129_2e4_ep400_20260129 \
--output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/axon_hm/brainmae-vitb-pretrain_32x256_patch16_w_dataset_token_ALL0_instanceMULTI_overlayvol1_rfg4096_Segments_SegNetData0129_2e4_ep400_20260129 \
--data_paths /media/hddshare/general/mae_brain/data/bmae/packed_training_data/fMOST \
--train_datas train_Seg_Net_0129 \
--mixture_weights 1 \
--point2segment \
--max_vol_num 1 \
--dataset_size 4096 \
--transforms rfg \
--wandb_project brain-mae-finetune-axon-hm \
--wandb_run_name mothra_axon_hm_rfg4096_Segments_SegNetData0129_2e4_ep400_20260129
