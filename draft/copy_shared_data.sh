#!/usr/bin/env bash
set -euo pipefail

# Run manually in tmux. Override these variables if a source lives elsewhere.
SOURCE_ROOT="${SOURCE_ROOT:-/media/hddshare/general/gxy/data/bmae}"
SOURCE_FMOST="${SOURCE_FMOST:-${SOURCE_ROOT}/packed_training_data/fMOST}"
SOURCE_AXON="${SOURCE_AXON:-${SOURCE_ROOT}/axon}"
SHARED_ROOT="${SHARED_ROOT:-/media/hddshare/general/mae_brain}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_ROOT="${SHARED_ROOT}/data/bmae"
PROJECT_ROOT="${SHARED_ROOT}/brain-mae"

dry_run=false
case "${1:-}" in
    "") ;;
    --dry-run) dry_run=true ;;
    *) echo "Usage: bash $0 [--dry-run]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

for tool in jq rsync blockfs-rebase realpath; do
    if ! command -v "$tool" >/dev/null; then
        echo "Required command not found: $tool" >&2
        if [[ "$tool" == jq ]]; then
            echo "Install the jq executable: conda install -c conda-forge jq" >&2
            echo "The pip package jq provides Python bindings, not this command." >&2
        fi
        exit 1
    fi
done

# Deduplicate images shared by multiple training/evaluation manifests.
declare -a sources=() destinations=()
declare -A queued=()
queue_copy() {
    local source destination
    source="$(realpath -m -- "$1")"
    destination="$(realpath -m -- "$2")"
    if [[ "$source" == "$destination" && "$source" == "${REPO_ROOT}/data/"* ]]; then
        return  # Manifests may already be in the shared checkout.
    fi
    if [[ "$source" == "$destination" || "$destination" == "$source/"* || "$source" == "$destination/"* ]]; then
        echo "Source and destination overlap: $source -> $destination" >&2
        exit 1
    fi
    if [[ -n "${queued[$destination]+present}" ]]; then
        [[ "${queued[$destination]}" == "$source" ]] || {
            echo "Conflicting sources for $destination" >&2; exit 1;
        }
        return
    fi
    queued["$destination"]="$source"
    sources+=("$source")
    destinations+=("$destination")
}

queue_relative_paths() {
    local source_root="$1" destination_root="$2" paths="$3" path
    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        if [[ "$path" == /* || "/$path/" == *"/../"* ]]; then
            echo "Expected a relative dataset path: $path" >&2
            exit 1
        fi
        queue_copy "$source_root/$path" "$destination_root/$path"
    done <<< "$paths"
}

# Full cell channels preserve coordinates and support MAE/DINO pretraining.
for manifest in "$REPO_ROOT"/data/*.json; do
    paths="$(jq -r '[.annotations[] as $a | $a.channels[] | $a.brain_id + "/" + .] | unique[]' "$manifest")"
    queue_relative_paths "$SOURCE_ROOT" "$IMAGE_ROOT" "$paths"
    queue_copy "$manifest" "$PROJECT_ROOT/data/$(basename "$manifest")"
done

# Neurite training, SNR evaluation, and held-out evaluation (including IDs 6/7).
for name in train_Seg_Net_0129 Seg_Net_data_for_different_SNR_0204 train_Seg_Net; do
    manifest="$REPO_ROOT/data/fMOST/$name.json"
    [[ -f "$manifest" ]] || manifest="$SOURCE_FMOST/$name.json"
    paths="$(jq -r '
        (if type == "array" then . else [.] end)[] as $brain |
        ($brain.annotation_dict | to_entries[]) as $axon |
        ([$brain.brain_ID, "1", $axon.value] | join("/")),
        ($brain.subvol_dict[$axon.key][] as $volume |
         if ($volume | contains("pretif")) then
             [$brain.brain_ID, $axon.key, $volume] | join("/")
         else
             [$brain.brain_ID, $volume] | join("/")
         end)' "$manifest")"
    queue_relative_paths "$SOURCE_FMOST" "$IMAGE_ROOT/packed_training_data/fMOST" "$paths"
    queue_copy "$manifest" "$IMAGE_ROOT/packed_training_data/fMOST/$name.json"
    queue_copy "$manifest" "$PROJECT_ROOT/data/fMOST/$name.json"
done

# Confocal neurite evaluation annotations contain their traced points directly.
manifest="$SOURCE_AXON/4x_Thy1_test_left.json"
paths="$(jq -r '(if type == "array" then . else [.] end)[] | .brain_ID + "/" + .marker_folder' "$manifest")"
queue_relative_paths "$SOURCE_AXON" "$IMAGE_ROOT/axon" "$paths"
queue_copy "$manifest" "$IMAGE_ROOT/axon/4x_Thy1_test_left.json"
queue_copy "$manifest" "$PROJECT_ROOT/data/axon/4x_Thy1_test_left.json"

# Check the entire plan before copying. Never rebase a source image tree.
image_root="$(realpath -m -- "$IMAGE_ROOT")"
for source_root in "$SOURCE_ROOT" "$SOURCE_FMOST" "$SOURCE_AXON"; do
    source_root="$(realpath -m -- "$source_root")"
    if [[ "$image_root" == "$source_root" || "$image_root" == "$source_root/"* || "$source_root" == "$image_root/"* ]]; then
        echo "Source and copied image trees must be separate: $source_root" >&2
        exit 1
    fi
done
for i in "${!sources[@]}"; do
    [[ -e "${sources[$i]}" ]] || { echo "Missing source: ${sources[$i]}" >&2; exit 1; }
    printf '%s -> %s\n' "${sources[$i]}" "${destinations[$i]}"
done
if "$dry_run"; then
    echo "Dry run: no data copied or rebased."
    exit 0
fi

for i in "${!sources[@]}"; do
    source="${sources[$i]}"
    destination="${destinations[$i]}"
    if [[ -d "$source" ]]; then
        mkdir -p "$destination"
        source+="/"
        destination+="/"
    else
        mkdir -p "$(dirname "$destination")"
    fi
    # Follow symlinks into real copies; retain partial transfers for reruns.
    rsync -rtL --partial --partial-dir=.rsync-partial --no-whole-file \
        --info=progress2 --human-readable -- "$source" "$destination"
done

# Rebase only the copied data, using the standard BlockFS command.
cd "$IMAGE_ROOT"
find . -name "precomputed.blockfs" -exec blockfs-rebase {} \;

mkdir -p "$PROJECT_ROOT/output_dir/pre-train/brainmae-vitb" "$PROJECT_ROOT/output_dir/axon_hm"
echo "Data copy finished. Model weights were not copied."
