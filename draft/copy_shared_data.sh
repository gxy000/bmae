#!/usr/bin/env bash
set -euo pipefail

# Run manually in tmux. Edit these paths, or override them in the environment.
SOURCE_ROOT="${SOURCE_ROOT:-/media/hddshare/general/gxy/data/bmae}"
SOURCE_FMOST="${SOURCE_FMOST:-${SOURCE_ROOT}/packed_training_data/fMOST}"
SOURCE_AXON="${SOURCE_AXON:-${SOURCE_ROOT}/axon}"
SHARED_ROOT="${SHARED_ROOT:-/media/hddshare/general/mae_brain}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_ROOT="${SHARED_ROOT}/data/bmae"
PROJECT_ROOT="${SHARED_ROOT}/brain-mae"
BRAIN="20240418_11_27_21_Xinyi_Thy1_EGFP_TH_Syto"

mkdir -p "$IMAGE_ROOT/$BRAIN" "$IMAGE_ROOT/packed_training_data/fMOST" "$IMAGE_ROOT/axon/$BRAIN"
mkdir -p "$PROJECT_ROOT/data/fMOST" "$PROJECT_ROOT/data/axon"
mkdir -p "$PROJECT_ROOT/output_dir/pre-train/brainmae-vitb" "$PROJECT_ROOT/output_dir/axon_hm"

# Cell training/evaluation and MAE/DINO pretraining: both complete image channels.
# All six retained EGFP variants use these same two channels.
rsync -rtL --partial --no-whole-file --info=progress2 \
    "$SOURCE_ROOT/$BRAIN/Ex_488_Em_0_precomputed" "$IMAGE_ROOT/$BRAIN/"
rsync -rtL --partial --no-whole-file --info=progress2 \
    "$SOURCE_ROOT/$BRAIN/Ex_642_Em_2_precomputed" "$IMAGE_ROOT/$BRAIN/"

# Neurite training and held-out evaluation, including axon IDs 6 and 7.
# Copy the complete dataset folder, including TIFF images and the SWC folder 1/.
rsync -rtL --partial --no-whole-file --info=progress2 \
    "$SOURCE_FMOST/train_Seg_Net" "$IMAGE_ROOT/packed_training_data/fMOST/"

# Neurite evaluation at different signal-to-noise levels: TIFFs and SWCs.
rsync -rtL --partial --no-whole-file --info=progress2 \
    "$SOURCE_FMOST/Seg_Net_data_for_different_SNR" "$IMAGE_ROOT/packed_training_data/fMOST/"

# Confocal neurite evaluation: the channel named in 4x_Thy1_test_left.json.
rsync -rtL --partial --no-whole-file --info=progress2 \
    "$SOURCE_AXON/$BRAIN/Ex_488_Em_0_precomputed" "$IMAGE_ROOT/axon/$BRAIN/"

# Annotation manifests already included in this repository.
# Skip this copy if the repository itself is already at the shared destination.
if [[ ! "$REPO_ROOT/data" -ef "$PROJECT_ROOT/data" ]]; then
    rsync -rtL --partial --info=progress2 "$REPO_ROOT/data/" "$PROJECT_ROOT/data/"
fi
cp "$REPO_ROOT/data/fMOST/train_Seg_Net_0129.json" "$IMAGE_ROOT/packed_training_data/fMOST/"
cp "$REPO_ROOT/data/fMOST/Seg_Net_data_for_different_SNR_0204.json" "$IMAGE_ROOT/packed_training_data/fMOST/"

# Additional evaluation manifests supplied with the original image datasets.
cp "$SOURCE_FMOST/train_Seg_Net.json" "$IMAGE_ROOT/packed_training_data/fMOST/"
cp "$SOURCE_AXON/4x_Thy1_test_left.json" "$IMAGE_ROOT/axon/"
cp "$SOURCE_FMOST/train_Seg_Net.json" "$PROJECT_ROOT/data/fMOST/"
cp "$SOURCE_AXON/4x_Thy1_test_left.json" "$PROJECT_ROOT/data/axon/"

# Rebase the copied BlockFS indexes.
cd "$IMAGE_ROOT"
find . -name "precomputed.blockfs" -exec blockfs-rebase {} \;

echo "Data copy finished. Model weights were not copied."
