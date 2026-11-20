"""Synthetic CPU checks for the restored training dependency paths."""

import contextlib
import importlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np
import torch


ENTRY_POINTS = (
    "main_pretrain",
    "main_pretrain_multidata",
    "main_pretrain_DINO",
    "main_finetune_hm_offset",
    "main_finetune_axon_hm_multidata",
)


def cpu_scaler(loss, optimizer, update_grad=True, **kwargs):
    loss.backward()
    if update_grad:
        optimizer.step()


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def small_model(self, module_name):
        module = importlib.import_module(module_name)
        cube_size = (
            32 if module_name == "models_mae" or "cn_hm_offset" in module_name else (32, 32, 32)
        )
        return module.MaskedAutoencoderViT(
            vox_size=cube_size,
            in_chans=2,
            embed_dim=48,
            depth=1,
            num_heads=4,
            decoder_embed_dim=32,
            decoder_depth=1,
            decoder_num_heads=4,
        )

    def run_epoch(self, engine, model, batch, criterion=None):
        args = SimpleNamespace(
            accum_iter=1,
            lr=1e-3,
            min_lr=0,
            warmup_epochs=0,
            epochs=1,
            mask_ratio=0.75,
            focalloss_pos=0.85,
            dist_weight=1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        before = model.patch_embed.proj.weight.detach().clone()
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch("torch.cuda.synchronize"))
            stack.enter_context(patch("torch.cuda.amp.autocast", contextlib.nullcontext))
            if criterion is None:
                metrics = engine.train_one_epoch(
                    model, [batch], optimizer, torch.device("cpu"), 0, cpu_scaler, args=args
                )
            else:
                metrics = engine.train_one_epoch(
                    model,
                    criterion,
                    [batch],
                    optimizer,
                    torch.device("cpu"),
                    0,
                    cpu_scaler,
                    args=args,
                )
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertFalse(torch.equal(before, model.patch_embed.proj.weight))

    def test_all_training_entry_points_import_and_expose_help(self):
        for name in ENTRY_POINTS:
            with self.subTest(entry=name):
                module = importlib.import_module(name)
                parser = module.get_args_parser()
                self.assertIn("--help", parser.format_help())
                args = parser.parse_args(["--epochs", "1"])
                self.assertEqual(args.epochs, 1)

    def test_default_models_and_multidata_input_configuration(self):
        import main_pretrain
        import main_pretrain_multidata
        import main_finetune_hm_offset
        import main_finetune_axon_hm_multidata
        import models_mae
        import models_mae_multidata
        import models_mae_cn_hm_offset
        import models_mae_axon_hm_multidata

        for entry, models in (
            (main_pretrain, models_mae),
            (main_pretrain_multidata, models_mae_multidata),
            (main_finetune_hm_offset, models_mae_cn_hm_offset),
            (main_finetune_axon_hm_multidata, models_mae_axon_hm_multidata),
        ):
            args = entry.get_args_parser().parse_args([])
            self.assertTrue(callable(getattr(models, args.model)))
        args = main_pretrain_multidata.get_args_parser().parse_args([])
        self.assertIsInstance(args.data_path, list)
        self.assertIsInstance(args.marker_folder, list)
        self.assertEqual(len(args.data_path), len(args.marker_folder))
        self.assertEqual(args.vol_size, 32)
        with patch.object(models_mae_multidata, "MaskedAutoencoderViT") as constructor:
            models_mae_multidata.__dict__[args.model]()
        self.assertEqual(constructor.call_args[1]["vox_size"], (32, 256, 256))
        self.assertEqual(constructor.call_args[1]["in_chans"], 1)

    def test_multidata_pretraining_resolves_folder_lists_and_rejects_mismatches(self):
        import util.brain_datasets_multidata as datasets

        with patch.object(datasets, "pretif_open", return_value=object()) as reader:
            dataset = datasets.BrainImageDataset(
                data_paths=["brain_a", "brain_b"],
                data_folder_lists=[None, ["marker_a", "marker_b"]],
            )
            self.assertEqual(len(dataset.pretif_list), 2)
            self.assertEqual(
                reader.call_args_list, [call("brain_a/marker_a"), call("brain_b/marker_b")],
            )
            with self.assertRaisesRegex(ValueError, "one folder name per data path"):
                datasets.BrainImageDataset(
                    data_paths=["brain_a", "brain_b"], data_folder_lists=[["marker_a"]]
                )

    def test_mae_pretraining_optimizer_steps(self):
        for module_name, engine_name in (
            ("models_mae", "engine_pretrain"),
            ("models_mae_multidata", "engine_pretrain_multidata"),
        ):
            with self.subTest(model=module_name):
                model = self.small_model(module_name)
                inputs = torch.rand(1, 2, 32, 32, 32)
                batch = (inputs, torch.zeros(1)) if "multidata" in module_name else inputs
                self.run_epoch(importlib.import_module(engine_name), model, batch)

    def test_cell_finetuning_optimizer_steps(self):
        model = self.small_model("models_mae_cn_hm_offset")
        inputs = torch.rand(1, 2, 32, 32, 32)
        heatmap = torch.zeros(1, 1, 16, 16, 16)
        heatmap[:, :, 8, 8, 8] = 1
        offsets = torch.zeros(1, 3, 16, 16, 16)
        engine = importlib.import_module("engine_finetune_cn_hm_offset")
        self.run_epoch(engine, model, (inputs, heatmap, offsets))
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch("torch.cuda.amp.autocast", contextlib.nullcontext))
            metrics = engine.evaluate(
                [(inputs, heatmap, offsets, [])], model, torch.device("cpu"), 1, 0, 0.85
            )
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_axon_finetuning_optimizer_steps(self):
        for module_name, engine_name in (
            ("models_mae_axon_hm_multidata", "engine_finetune_axon_hm_multidata"),
        ):
            with self.subTest(model=module_name):
                module = importlib.import_module(module_name)
                model = self.small_model(module_name)
                heatmap = torch.zeros(16, 16, 16)
                heatmap[8, 8, 6:10] = 1
                target = {"heatmap": heatmap}
                if "multidata" in module_name:
                    target["instance_map"] = torch.full((16, 16, 16), -1, dtype=torch.long)
                    target["instance_map"][8, 8, 6:10] = 1
                batch = (torch.rand(1, 2, 32, 32, 32), [target], torch.zeros(1, 1))
                self.run_epoch(
                    importlib.import_module(engine_name),
                    model,
                    batch,
                    criterion=module.SetCriterion(),
                )

    def test_dino_multicrop_loss_backpropagates(self):
        import models_dino

        encoder = models_dino.BrainDINOEncoder(
            vox_size=32, in_chans=2, embed_dim=48, depth=1, num_heads=4
        )
        model = models_dino.BrainDINO(
            encoder, out_dim=16, head_hidden_dim=32, head_bottleneck_dim=8
        )
        crops = [torch.rand(1, 2, 32, 32, 32) for _ in range(2)]
        crops.append(torch.rand(1, 2, 16, 16, 16))
        student = model(crops)
        with torch.no_grad():
            teacher = model(crops[:2])
        criterion = models_dino.DINOLoss(
            out_dim=16,
            ncrops=3,
            warmup_teacher_temp=0.04,
            teacher_temp=0.04,
            warmup_teacher_temp_epochs=0,
            nepochs=1,
        )
        loss = criterion(student, teacher, epoch=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(encoder.patch_embed.proj.weight.grad)
        self.assertTrue(torch.isfinite(encoder.patch_embed.proj.weight.grad).all())

    def test_fmost_training_uses_loaded_volume_shape(self):
        import util.axon_seg_datasets_hm_multidata as datasets

        dataset = datasets.AxonSegDataset.__new__(datasets.AxonSegDataset)
        subvol = "subvol_0_8_0_8_0_8_pretif"
        dataset.samples = [
            {
                "modality": "fMOST",
                "brain_id": "brain",
                "axon_key": "axon",
                "swc_path": "unused.swc",
                "subvol_folder": subvol,
                "offset": [0, 0, 0],
                "ntype": [3, 4],
            }
        ]
        dataset._tif_cache = {("brain", "axon", subvol): np.ones((8, 8, 8))}
        dataset.max_vol_num = 1
        dataset.subvol_size = np.array([4, 4, 4])
        dataset.crop_vol_size = dataset.subvol_size
        dataset.R = 2
        dataset.im_channels = 1
        dataset.labeled = True
        dataset.instance = True
        dataset.twochannel = False
        dataset.point2segment = False
        dataset.transforms = ["whole_volume"]
        dataset.params = {}
        points = {0: np.array([[1, 1, 1], [1, 1, 2]])}
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(datasets, "_safe_load_morphology", return_value=object())
            )
            fetch = stack.enter_context(patch.object(dataset, "fetch_points", return_value=points))
            stack.enter_context(
                patch.object(
                    dataset,
                    "random_sample",
                    return_value=(np.ones((1, 4, 4, 4)), points, 0, 0, 0, False),
                )
            )
            volume, target, _ = dataset[0]
        self.assertEqual(fetch.call_args[0][2], (8, 8, 8))
        self.assertEqual(tuple(volume.shape), (1, 4, 4, 4))
        self.assertEqual(tuple(target["heatmap"].shape), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
