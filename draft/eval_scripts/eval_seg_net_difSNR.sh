#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."


CKPT_FILE="/media/hddshare/general/mae_brain/brain-mae/output_dir/axon_hm/brainmae-vitb-scratch_32x256_patch16_w_dataset_token_ALL0_instanceMULTI_overlayvol1_rfg4096_Seg_Net_0129_2e4_ep400_20260203/checkpoint-399.pth"
MODEL_NAME="axonseg_hm_dense_fromscratch_SegNetData0129_2e4_ep400_20260129"

CUDA_VISIBLE_DEVICES=0 python test_axonseg_multidata_wholev.py \
    --data_path /media/hddshare/general/mae_brain/data/bmae/packed_training_data/fMOST/ \
    --testdata Seg_Net_data_for_different_SNR_0204 \
    --labeled \
    --fMOST \
    --hm_threshold 0.5 \
    --merge_dist 4 \
    --eval_radius 0 1 2 3 4 5 6 7 8 9 10 \
    --overlap 8 32 32 \
    --checkpoints "${CKPT_FILE}" \
    --model_archs mae_vit_base_patch16_1c_32 \
    --models "${MODEL_NAME}" \
    --plot_labels "${MODEL_NAME}" \
    --thick_viz
