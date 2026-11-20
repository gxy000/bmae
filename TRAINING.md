# Pretraining and finetuning

Training is retained alongside evaluation. MAE and DINO learn encoders from
unlabeled volumes; the cell and axon finetuning entry points load those encoder
weights through `--finetune`. Finetuning also supports training from scratch by
omitting `--finetune`, and resuming a training run through `--resume`.

Activate the environment prepared following [README.md](README.md):

```bash
conda activate brain-mae
```

Prepare the shared dataset copy with `bash draft/copy_shared_data.sh`
(see [README.md](README.md)). Launchers work from any current directory:

| Workflow | Launcher |
| --- | --- |
| Cell MAE pretraining | `draft/train_scripts/launch_pretrain.sh` |
| Cell DINO pretraining | `draft/train_scripts/celldet/launch_pretrain_dino.sh` |
| Cell finetuning | `draft/train_scripts/celldet/launch_finetune.sh` |
| Neurite MAE pretraining | `draft/train_scripts/neuriteseg/launch_pretrain.sh` |
| Neurite finetuning | `draft/train_scripts/neuriteseg/launch_finetune.sh` |

The examples below use direct `torchrun` invocation from the repository root.

| Workflow | Entry point | Purpose |
| --- | --- | --- |
| `pretrain_celldet` | `main_pretrain.py` | Single-dataset MAE pretraining, including paired nuclei/marker input |
| `pretrain_axon` | `main_pretrain_multidata.py` | MAE pretraining with dataset tokens, including 32 × 256 × 256 axon inputs |
| `pretrain_dino` | `main_pretrain_DINO.py` | DINO pretraining and compatible cell-encoder checkpoint export |
| `finetune_celldet` | `main_finetune_hm_offset.py` | Cell heatmap/offset finetuning |
| `finetune_axon` | `main_finetune_axon_hm_multidata.py` | Axon heatmap and instance-embedding finetuning |

Cell finetuning uses only `models_mae_cn_hm_offset.py`, matching
`test_evaluate_cn_wholev.py`. Axon finetuning uses
`models_mae_axon_hm_multidata.py`, matching the neurite checkpoints selected by
the evaluation launchers.

Multi-dataset MAE pretraining remains available for axon encoders. Cell
multi-dataset finetuning has been removed. The pretraining methods can be used
with compatible downstream input channels and encoder architectures; model
aliases and checkpoint parameter names are preserved.

For all options:

```bash
python main_pretrain.py --help
python main_finetune_hm_offset.py --help
python main_finetune_axon_hm_multidata.py --help
```

The examples use the shared lab image and checkpoint roots. The copy script
prepares image data; checkpoint files are assumed to be supplied separately.
Choose batch sizes for the available GPUs.

Paired-channel MAE pretraining for cell detection:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main_pretrain.py \
  --model mae_vit_base_patch16 \
  --data_path /media/hddshare/general/mae_brain/data/bmae/20240418_11_27_21_Xinyi_Thy1_EGFP_TH_Syto \
  --nuclei_folder Ex_642_Em_2_precomputed \
  --marker_folder Ex_488_Em_0_precomputed \
  --batch_size 8 --accum_iter 2 --mask_ratio 0.75 \
  --epochs 800 --warmup_epochs 40 --blr 1.5e-4 \
  --output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/pretrain_celldet
```

Single-channel MAE pretraining for axon inputs on the copied EGFP images:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main_pretrain_multidata.py \
  --model mae_vit_base_patch16_1c_32 --vol_size 32 \
  --data_path /media/hddshare/general/mae_brain/data/bmae/20240418_11_27_21_Xinyi_Thy1_EGFP_TH_Syto \
  --marker_folder Ex_488_Em_0_precomputed \
  --batch_size 8 --accum_iter 2 --mask_ratio 0.75 \
  --epochs 800 --warmup_epochs 40 --blr 1.5e-4 \
  --output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/pretrain_axon
```

The neurite pretraining example is for exploring encoder training. The original
neurite finetuning launcher loads the separately supplied NeuN/GFAP checkpoint;
the example does not reproduce that checkpoint. Pretraining and finetuning
remain separate commands with explicitly selected checkpoint paths.

Cell finetuning:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main_finetune_hm_offset.py \
  --model mae_vit_base_patch16_4 \
  --finetune /media/hddshare/general/mae_brain/brain-mae/output_dir/pre-train/brainmae-vitb/pretrain_Syto_EGFP_patch16_0602_checkpoint-799.pth \
  --data_path /media/hddshare/general/mae_brain/data/bmae \
  --train_data data/EGFP_200_3x512.json \
  --batch_size 8 --accum_iter 2 --epochs 400 --blr 5e-4 \
  --R 2 --focalloss_pos 0.85 --transforms rfg \
  --output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/finetune_celldet --log_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/finetune_celldet
```

Axon instance/heatmap finetuning:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main_finetune_axon_hm_multidata.py \
  --model mae_vit_base_patch16_1c_32 \
  --finetune /media/hddshare/general/mae_brain/brain-mae/output_dir/pre-train/neun_gfap_patch16_w_dataset_token_ALL0_32x256_251030_h100s_checkpoint-799.pth \
  --data_paths /media/hddshare/general/mae_brain/data/bmae/packed_training_data/fMOST --train_datas train_Seg_Net_0129 \
  --batch_size 8 --accum_iter 2 --epochs 400 --blr 2e-4 \
  --threshold_pos 0.5 --point2segment --max_vol_num 1 \
  --dataset_size 4096 --transforms rfg \
  --output_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/finetune_axon --log_dir /media/hddshare/general/mae_brain/brain-mae/output_dir/finetune_axon
```

All training engines, loss functions, dataset classes, optimizer scheduling,
distributed helpers, checkpoint loading/saving, and logging dependencies used by
these entry points are included. `wandb` and TensorBoard are included in the
dependency manifest; logging is controlled by the original CLI options.

Synthetic checks exercise optimizer steps for both MAE pretraining variants,
the supported cell finetuning and neurite heatmap/instance models. They also
check DINO gradients, the fMOST loader, and the evaluation regressions:

```bash
MPLBACKEND=Agg python -B -m unittest discover -s tests -v
```

Production training and distributed GPU runs are not launched by these checks.

