# Supported training and evaluation code

The shared copy retains two downstream models:

| Task | Evaluator | Model module |
| --- | --- | --- |
| Cell detection | `test_evaluate_cn_wholev.py` | `models_mae_cn_hm_offset.py` |
| Neurite segmentation | `test_axonseg_multidata_wholev.py` | `models_mae_axon_hm_multidata.py` |

Five training entry points remain: cell MAE, multidata MAE, DINO, cell
heatmap/offset finetuning, and neurite heatmap/instance finetuning. See
[TRAINING.md](TRAINING.md) for the commands. Cell multidata, fixed-size sampling,
four-rank reproduction and axon heatmap-only entry points/models are excluded.
The evaluator and regression tests follow these retained workflows.

The five evaluation launchers are under `draft/eval_scripts/`; cell and neurite
finetuning launchers are under `draft/train_scripts/celldet/` and
`draft/train_scripts/neuriteseg/`. Dataset and checkpoint defaults now use
`/media/hddshare/general/mae_brain/`, as do training and evaluation outputs.
Model architecture, losses, seeds, optimizer settings and checkpoint basenames
are preserved. In particular, the original cell offset-loss broadcasting stays
unchanged.

`draft/copy_shared_data.sh` prepares independent image and annotation copies,
including the six requested EGFP variants and neurite training/evaluation data.
It copies symlink targets, rebases BlockFS indexes and supports resumed rsync
transfers. Checkpoints are assumed to be supplied separately. The script has
not been run on the real datasets. It uses Bash, jq and rsync for copying, then
`find . -name "precomputed.blockfs" -exec blockfs-rebase {} \;` from the copied
image root. The Python planner and custom BlockFS rewrite code were removed.

Git metadata, obsolete model families, unrelated inference code and unused
evaluation metrics/renderers remain removed. The original license is retained
in [LICENSE](LICENSE). Removed tests for the deleted reproduction workflow and
the source snapshot from shared-path preparation are archived outside this
repository. Current dataset filenames and shared locations are in
[README.md](README.md).

The repository review removed the two remaining unused runtime modules:
`util/axon_seg_datasets_hmonly_multidata.py` and
`util/cell_detection_datasets_cn_hm_offset_sampling.py`. All 28 remaining runtime
Python files are reachable from the seven supported training/evaluation entry
points. The two Python files under `tests/` remain for verification.

The review also fixed model defaults that referred to removed architectures,
multidata pretraining's list defaults and crop-shape selection, and launchers
that depended on the current directory. Neurite checkpoint loading now selects
the architecture independently of the checkpoint filename. Dead branches for
the deleted heatmap-only model and obsolete dataset debug programs were removed.
Model definitions and training engines are unchanged, including the original
cell offset-loss broadcasting. All ten retained annotation files are unchanged.

There are five training launchers and five evaluation launchers. Their twelve
emitted commands parse with the retained entry points, models, and shared paths
when invoked from outside the repository. All Bash scripts pass syntax checks,
and the training/evaluation code passes 20 CPU regression tests. Python static
checks pass. Setup manifests and a Bash environment setup helper are included;
a fresh installation, production GPU runs, dataset copying, and BlockFS rebasing
have not been executed.

The source snapshot, removed modules, dependency graph, integrity checks,
launcher checks, and test log for this review are in
`../brain-mae-cleanup-audit/repo-review-20261120-29a80_5a/`.
Earlier shared-path preparation is recorded in
`../brain-mae-cleanup-audit/sharing-20261120-97tgq661/`.

Recovery and audit files are outside this directory:

- `../brain-mae-cleanup-audit/supported-models-20261106/source-before.tar.gz`:
  source snapshot immediately before narrowing model support. Large preserved
  data, figure, and checkpoint payloads are excluded.
- `../brain-mae-cleanup-audit/supported-models-20261106/removed.json`: the seven
  files removed in this update.
- `../brain-mae-cleanup-audit/supported-models-20261106/shell-checks.json`:
  retained evaluation invocation counts.
- `../brain-mae-cleanup-audit/changes.json`: current removal inventory.
- `../brain-mae-cleanup-audit/dependencies-after.json`: current dependency graph.
- `../brain-mae-source-before-cleanup.tar.gz`: source recovery archive from the
  original cleanup. Earlier audit files record the initial evaluation cleanup
  and subsequent restoration of training.

## Data-folder cleanup (2026-11-20)

The ten retained manifests are listed in [README.md](README.md). They cover
training for all six requested EGFP comparison models (`egfp_v1_400`,
`egfp_v2_400`, `egfp_v3_400`, `egfp_v4_400`, `egfp_v5_400`, `egfp_v9_400`), shared
cell validation and evaluation, and neurite training/validation and different-SNR
evaluation. The active neurite scratch and pretrained checkpoints use
`train_Seg_Net_0129`. Pretraining reads external volumes rather than local
annotation manifests.

The local folder was reduced from 296 files (889,325,928 bytes) to ten files
(90,645,352 bytes). The other 286 files comprise older dataset variants, duplicate
manifests, alignment experiments, 9x data from the removed evaluator, generated
predictions, and visualization previews. All ten retained files are unchanged.
Cell training and evaluation defaults now select the retained EGFP manifests;
explicit launcher arguments and model/loss behavior are unchanged.

Recovery and audit files are outside this repository at
`../brain-mae-cleanup-audit/data-20261120-0dszi3iz/`:

- `inventory-before.json`: original paths, SHA-256 hashes, sizes, initial
  retention decisions, and external manifest dependencies.
- `inventory-current.json`: updated retention decisions covering all six EGFP
  comparison models, their training manifests and original training launchers.
- `removed-data/`: the 286 removed files, with their original paths beneath
  `data/`. Copy files back under `data/` to restore them.
- `source-before-*`: snapshots of the updated source and documentation files.
- `verification.json`: retained-file integrity, dataset loading and regression
  test results.

External raw datasets, SWCs, training/evaluation manifests, checkpoint files and
pretraining volumes are untouched. The external evaluation manifests
`train_Seg_Net.json` and `4x_Thy1_test_left.json` had no local copies before this
cleanup. Their external availability was not verified because a mounted-data
inspection blocked; they remain required by the current neurite launchers.
