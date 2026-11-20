# Brain MAE training and evaluation

This directory supports pretraining, finetuning, and evaluation for cell
detection and axon models. See [TRAINING.md](TRAINING.md) for the retained training
workflows and launch examples. The evaluation launchers are in
`draft/eval_scripts`.

New lab members can start with [REPO_GUIDE.html](REPO_GUIDE.html), a standalone
HTML introduction to the code structure, data, and first experiments.

| Launchers | Local entry point | Model modules |
| --- | --- | --- |
| `eval_celldet_egfp_4x*.sh` | `test_evaluate_cn_wholev.py` | `models_mae_cn_hm_offset.py` |
| `eval_axon_4x.sh`, `eval_seg_net_*.sh` | `test_axonseg_multidata_wholev.py` | `models_mae_axon_hm_multidata.py` |

Supported downstream models are defined by these two evaluators. Cell detection
uses only `models_mae_cn_hm_offset.py`. The axon evaluator uses the retained heatmap/instance model.

Create the Linux environment and install the pinned dependencies:

```bash
conda env create -f environment.yml
conda activate brain-mae
bash draft/setup_environment.sh
```

The setup script installs the lab's volume readers from pinned public GitHub
revisions and applies the `timm==0.3.2` import compatibility fix for PyTorch 1.12.
The environment follows the tested Python 3.7 / CUDA 11.3 stack; GPU runs require
a compatible NVIDIA driver. An existing working `mae_viz` environment can also
be used. A fresh environment installation has not been run during this review.

After preparing the dataset copy and supplying the checkpoints, run a launcher:

```bash
bash draft/eval_scripts/eval_axon_4x.sh
```

Launchers resolve the repository root themselves, so they can also be invoked by
absolute path from another directory. Review their dataset, checkpoint, and GPU
settings before a full run. Dataset, checkpoint and evaluation output paths use the shared lab root
`/media/hddshare/general/mae_brain/`.
Run any local Python entry point with `--help` to inspect its supported options.

Dependencies are recorded in [requirements.txt](requirements.txt), including
the image readers, morphology libraries, TensorBoard, and W&B logging.

Cell detection reads `data/<testdata>.json`. Axon segmentation evaluation reads
its JSON from the selected external `--data_path`. The local `data/` folder keeps
the manifests for the current evaluation runs, the six EGFP comparison models
(`egfp_v1_400`, `egfp_v2_400`, `egfp_v3_400`, `egfp_v4_400`, `egfp_v5_400`,
`egfp_v9_400`), and their training/validation data:

| Local manifest | Used for |
| --- | --- |
| `data/EGFP_200_2_sampled1x512.json` | `egfp_v1_400` training: one volume |
| `data/EGFP_200_2x512.json` | `egfp_v2_400` training: two volumes |
| `data/EGFP_200_3x512.json` | `egfp_v3_400` training: three volumes |
| `data/EGFP_200_4x512.json` | `egfp_v4_400` training: four volumes |
| `data/EGFP_200_5x512_0831.json` | `egfp_v5_400` training: five-volume set |
| `data/EGFP_200_v9x512_0908.json` | `egfp_v9_400` training: nine volumes |
| `data/EGFP_200_val.json` | Cell training validation |
| `data/EGFP_200_eval4_0915.json` | Cell evaluation |
| `data/fMOST/train_Seg_Net_0129.json` | Neurite training and validation for the pretrained and scratch checkpoints |
| `data/fMOST/Seg_Net_data_for_different_SNR_0204.json` | Neurite evaluation across SNR levels |

Neurite launchers read manifests from the copied image directories. The copy
script obtains the two additional evaluation manifests, `train_Seg_Net.json`
(including held-out axons 6 and 7) and `4x_Thy1_test_left.json`, from the original
dataset roots. Their raw TIFFs, SWCs and precomputed volumes are included.
Older local annotations and generated previews remain in the recovery folder
recorded in [CLEANUP.md](CLEANUP.md).

To prepare the separate dataset copy, run this manually inside tmux from the
repository root (Bash 4+, `jq`, `rsync`, and `blockfs-rebase` are required):

```bash
bash draft/copy_shared_data.sh
```

The script copies full cell channels for all retained EGFP manifests and MAE/DINO
pretraining, plus the neurite training and evaluation data. It validates sources
before transferring, deduplicates repeated paths, follows symlinks into real
copies, and rebases copied BlockFS indexes with the standard command:
`find . -name "precomputed.blockfs" -exec blockfs-rebase {} \;`
from the copied image root. Rerunning the
command resumes partial transfers. Sources are read only; destination files are
updated without deleting unrelated files. `--dry-run` prints the validated plan
without copying or rebasing. If datasets live on different disks, edit the
source variables at the top of the script or set `SOURCE_ROOT`, `SOURCE_FMOST`
and `SOURCE_AXON` in the environment.

| Shared location | Contents |
| --- | --- |
| `/media/hddshare/general/mae_brain/data/bmae/` | Cell images and pretraining channels |
| `/media/hddshare/general/mae_brain/data/bmae/packed_training_data/fMOST/` | Neurite training, held-out evaluation and SNR datasets with manifests |
| `/media/hddshare/general/mae_brain/data/bmae/axon/` | Confocal neurite evaluation data and manifest |
| `/media/hddshare/general/mae_brain/brain-mae/data/` | Copies of the annotation manifests |
| `/media/hddshare/general/mae_brain/brain-mae/output_dir/` | Model weights, training logs and future checkpoints |
| `/media/hddshare/general/mae_brain/brain-mae/eval/` | Evaluation metrics and visualizations |

The script does not copy weights or launch training/evaluation. Checkpoint paths
assume weights will be placed under the shared output directory, including
`pre-train/brainmae-vitb/pretrain_Syto_EGFP_patch16_0602_checkpoint-799.pth`.
Dataset coordinates, model settings, checkpoint basenames and experiment
subdirectories are retained. The dataset copy has not been executed here.

Run the synthetic CPU training and evaluation checks without loading production
datasets or checkpoints:

```bash
MPLBACKEND=Agg python -B -m unittest discover -s tests -v
```

See [CLEANUP.md](CLEANUP.md) for the removal scope and validation results.
The original source license is in [LICENSE](LICENSE).
