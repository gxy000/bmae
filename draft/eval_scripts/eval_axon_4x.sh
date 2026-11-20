#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."


CKPT_FILE="/media/hddshare/general/mae_brain/brain-mae/output_dir/axon_hm/brainmae-vitb-scratch_32x256_patch16_w_dataset_token_ALL0_instanceMULTI_overlayvol1_rfg4096_Seg_Net_0129_2e4_ep400_20260203/checkpoint-399.pth"
MODEL_NAME="axonseg_hm_dense_fromscratch_SegNetData0129_2e4_ep400_20260129"

hm_thresholds=(0.5) # 0.5) # 0.2) # confocal

for hm_threshold in ${hm_thresholds[@]}; do
  echo "Evaluating for hm threshold: ${hm_threshold}"

  CUDA_VISIBLE_DEVICES=0 python test_axonseg_multidata_wholev.py \
    --data_path /media/hddshare/general/mae_brain/data/bmae/axon \
    --testdata 4x_Thy1_test_left \
    --labeled \
    --hm_threshold ${hm_threshold} \
    --merge_dist 4 \
    --p_min 1 --p_max 99 \
    --exclude_nuclei 10 \
    --eval_radius 0 1 2 3 4 5 6 7 8 9 10 \
    --match_cost_thr 5 \
    --min_overlap_length 5 \
    --match_tau_p 0.7 \
    --match_tau_g 0.7 \
    --overlap 8 32 32 \
    --instance_id_R 1 \
    --break_branch \
    --min_seg_length 15 \
    --checkpoints "${CKPT_FILE}" \
    --model_archs mae_vit_base_patch16_1c_32 \
    --models "${MODEL_NAME}" \
    --plot_labels "${MODEL_NAME}" \
    --thick_viz

  CUDA_VISIBLE_DEVICES=0 python test_axonseg_multidata_wholev.py \
    --data_path /media/hddshare/general/mae_brain/data/bmae/packed_training_data/fMOST \
    --testdata train_Seg_Net \
    --labeled \
    --fMOST --selected_axon_id 6 7 \
    --hm_threshold ${hm_threshold} \
    --merge_dist 4 \
    --p_min 1 --p_max 99 \
    --exclude_nuclei 10 \
    --eval_radius 0 1 2 3 4 5 6 7 8 9 10 \
    --match_cost_thr 5 \
    --min_overlap_length 5 \
    --match_tau_p 0.7 \
    --match_tau_g 0.7 \
    --overlap 8 32 32 \
    --instance_id_R 1 \
    --break_branch \
    --min_seg_length 15 \
    --checkpoints "${CKPT_FILE}" \
    --model_archs mae_vit_base_patch16_1c_32 \
    --models "${MODEL_NAME}" \
    --plot_labels "${MODEL_NAME}" \
    --viz_z_step 64 \
    --thick_viz

done
