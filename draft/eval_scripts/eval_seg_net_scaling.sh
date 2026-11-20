#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CKPT_FILEs=("/media/hddshare/general/mae_brain/brain-mae/output_dir/axon_hm/brainmae-vitb-pretrain_32x256_patch16_w_dataset_token_ALL0_instanceMULTI_overlayvol1_tpos05_rfg4096_Segments_SegNetData0129_2e4_ep400_20260401/checkpoint-399.pth")
MODEL_NAMEs=("axonseg_hm_dense_tpos05_SegNetData0129_2e4_ep400_20260129")

for i in ${!CKPT_FILEs[@]}; do
  CKPT_FILE="${CKPT_FILEs[$i]}"
  MODEL_NAME="${MODEL_NAMEs[$i]}"
  echo "Evaluating for model: ${MODEL_NAME}"
  CUDA_VISIBLE_DEVICES=1 python test_axonseg_multidata_wholev.py \
    --data_path /media/hddshare/general/mae_brain/data/bmae/packed_training_data/fMOST \
    --testdata train_Seg_Net \
    --labeled \
    --fMOST --selected_axon_id 6 7 \
    --p_min 1 --p_max 99 \
    --hm_threshold 0.5 \
    --merge_dist 4 \
    --eval_radius 0 1 2 3 4 5 6 7 8 9 10 \
    --match_cost_thr 5 \
    --min_overlap_length 5 \
    --match_tau_p 0.7 \
    --match_tau_g 0.7 \
    --overlap 8 32 32 \
    --instance_id_R 0.8 \
    --break_branch \
    --min_seg_length 15 \
    --checkpoints "${CKPT_FILE}" \
    --model_archs mae_vit_base_patch16_1c_32 \
    --models "${MODEL_NAME}" \
    --plot_labels "${MODEL_NAME}" \
    --thick_viz

  CUDA_VISIBLE_DEVICES=1 python test_axonseg_multidata_wholev.py \
    --data_path /media/hddshare/general/mae_brain/data/bmae/packed_training_data/fMOST \
    --testdata Seg_Net_data_for_different_SNR_0204 \
    --labeled \
    --fMOST \
    --p_min 1 --p_max 99 \
    --hm_threshold 0.5 \
    --merge_dist 4 \
    --eval_radius 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --overlap 8 32 32 \
    --checkpoints "${CKPT_FILE}" \
    --model_archs mae_vit_base_patch16_1c_32 \
    --models "${MODEL_NAME}" \
    --plot_labels "${MODEL_NAME}" \
    --thick_viz

done
