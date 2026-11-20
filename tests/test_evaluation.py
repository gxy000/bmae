"""CPU regression checks for the retained evaluation workflows.

Run with: python -B -m unittest discover -s tests -v
"""

import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import test_axonseg_multidata_wholev as instance
import test_evaluate_cn_wholev as cells
from util.bmae_utils import match_cell
from util.cell_detection_datasets_cn_hm_offset import _cn_decode_one_cls


class EvaluationTests(unittest.TestCase):
    def test_neurite_relative_checkpoint_uses_architecture_and_checkpoint_root(self):
        args = SimpleNamespace(
            models=["experiment"],
            model_archs=["mae_vit_base_patch16_1c_32"],
            plot_labels=["example"],
            checkpoints=["checkpoint-399.pth"],
            ckpt_root="/shared/output_dir/axon_hm",
            encoder="vitb",
        )
        model = MagicMock()
        model.to.return_value = model
        with patch.object(
            instance.models_mae_axon_hm, "mae_vit_base_patch16_1c_32", return_value=model
        ), patch.object(instance.torch, "load", return_value={"model": {}}) as load:
            result = instance._load_models(args, torch.device("cpu"))
        self.assertEqual(result, [model])
        load.assert_called_once_with(
            Path(args.ckpt_root) / "brainmae-vitb-experiment/checkpoint-399.pth",
            map_location="cpu",
        )
        model.load_state_dict.assert_called_once_with({}, strict=True)
        model.eval.assert_called_once_with()

    def test_neurite_model_lists_must_match_before_loading(self):
        args = SimpleNamespace(models=["one"], model_archs=["one", "two"], plot_labels=["one"])
        with self.assertRaisesRegex(ValueError, "matching"):
            instance._load_models(args, torch.device("cpu"))

    def test_cell_decoder_numpy_and_tensor_offsets(self):
        heat = np.zeros((1, 5, 5, 5), dtype=np.float32)
        heat[0, 2, 2, 2] = 0.9
        offsets = np.zeros((3, 5, 5, 5), dtype=np.float32)
        offsets[:, 2, 2, 2] = [0.25, -0.5, 0.75]
        expected = [[2.25, 1.5, 2.75, 0.9]]
        for heat_input, offset_input in (
            (heat, offsets),
            (torch.from_numpy(heat[None]), torch.from_numpy(offsets[None])),
        ):
            with self.subTest(tensor=torch.is_tensor(heat_input)):
                result = _cn_decode_one_cls(
                    heat_input, reg=offset_input, threshold_heat=0.5, exclude_border=False
                )
                np.testing.assert_allclose(result, expected)

    def test_cell_matching_does_not_mutate_inputs(self):
        prediction = np.array([[1, 2, 3], [8, 8, 8]])
        truth = np.array([[1, 2, 3], [4, 4, 4]])
        prediction_before, truth_before = prediction.copy(), truth.copy()
        tp, fn, fp = match_cell(prediction, truth, R=0)
        self.assertEqual([len(tp), len(fn), len(fp)], [1, 1, 1])
        np.testing.assert_array_equal(prediction, prediction_before)
        np.testing.assert_array_equal(truth, truth_before)
        self.assertEqual([len(x) for x in match_cell([], truth, R=0)], [0, 2, 0])

    def test_thresholds_preserve_per_model_configuration(self):
        self.assertEqual(cells._resolve_threshold_hms([0.3], 2), ([0.3, 0.3], False))
        self.assertEqual(cells._resolve_threshold_hms([0.3, 0.4], 2), ([0.3, 0.4], True))
        with self.assertRaises(ValueError):
            cells._resolve_threshold_hms([0.2, 0.3, 0.4], 2)
        self.assertEqual(instance._parse_eval_radii(["0,1", "1", "2"]), [0, 1, 2])
        with self.assertRaises(ValueError):
            instance._parse_eval_radii(["-1"])

    def test_patch_padding_preserves_small_volume(self):
        volume = np.arange(8).reshape(1, 2, 2, 2)
        patch = instance._extract_patch_with_tail_pad(volume, (0, 0, 0), (4, 4, 4))
        self.assertEqual(patch.shape, (1, 4, 4, 4))
        np.testing.assert_array_equal(patch[:, :2, :2, :2], volume)
        self.assertEqual(patch[0, -1, -1, -1], 0)

    def test_overlap_consensus_ignores_uncovered_voxels(self):
        positive = np.array([0, 1, 1, 2], dtype=np.uint16)
        covered = np.array([0, 1, 2, 2], dtype=np.uint16)
        result = instance._build_overlap_consensus_mask(positive, covered)
        np.testing.assert_array_equal(result, [False, True, False, True])

    def test_centerline_distance_and_empty_predictions(self):
        gt = {"centers": np.array([[0, 0, 0], [0, 0, 1]])}
        target = {"gt_skeleton_points": gt["centers"]}
        perfect = instance.esa(gt, target, distance_threshold=0)
        self.assertEqual(perfect["esa_mean"], 0)
        self.assertEqual(perfect["pred_within_r"], 1)
        self.assertEqual(perfect["gt_within_r"], 1)
        empty = instance.esa({"centers": []}, target, distance_threshold=1)
        self.assertEqual(empty["num_pred"], 0)
        self.assertEqual(empty["num_gt"], 2)

    def test_instance_matching_perfect_and_missing_predictions(self):
        line = np.array([[3, 3, 3], [3, 3, 4], [3, 3, 5]], dtype=float)
        for prediction, expected in (({1: line}, (1, 0, 0)), ({}, (0, 0, 1))):
            result = instance.evaluate_instance_matching_metrics_merge_and_split(
                prediction, {7: line}, r=0.5, cost_thr=0.5, tau_p=0.7, tau_g=0.7
            )
            for mode in ("one_to_one_pr", "merge_allowed", "split_allowed"):
                with self.subTest(mode=mode, empty=not prediction):
                    self.assertEqual(
                        tuple(result[mode][key] for key in ("tp", "fp", "fn")), expected
                    )

    def test_connectivity_metrics_detect_a_broken_line(self):
        line = np.array([[3, 3, x] for x in range(12)], dtype=float)
        target = {"gt_skeleton_points": line}
        for points, perfect in ((line, True), (line[[0, 1, 10, 11]], False)):
            with self.subTest(perfect=perfect):
                counts = instance.r_tolerant_erl_and_path_connectivity(
                    {"centers": points},
                    target,
                    radius_items=[("0", 0)],
                    path_length_items=[("5", 5)],
                )
                result = instance._finalize_r_tolerant_metric_counts(counts, [5])
                erl = result["r_tolerant_erl"]["normalized"]["0"]
                connectivity = result["r_tolerant_path_connectivity"]["score"]["0"]["5"]
                if perfect:
                    self.assertAlmostEqual(erl, 1)
                    self.assertAlmostEqual(connectivity, 1)
                else:
                    self.assertLess(erl, 1)
                    self.assertEqual(connectivity, 0)

    def test_heatmap_centerline_handles_empty_and_positive_masks(self):
        mask = np.zeros((12, 12, 12), dtype=bool)
        empty = instance._mask_to_centerline_pred(mask)
        self.assertEqual(empty["centers"].shape, (0, 3))
        mask[6, 6, 2:10] = True
        result = instance._mask_to_centerline_pred(mask)
        points = result["centers"]
        self.assertGreater(len(points), 0)
        self.assertTrue(np.all((points >= 0) & (points < 12)))
        self.assertEqual(len(points), len(result["inst_ids"]))

    def test_retained_models_run_on_cpu_and_reload_weights(self):
        torch.set_num_threads(1)
        inputs = torch.zeros(1, 2, 32, 32, 32)
        for name in (
            "models_mae_cn_hm_offset",
            "models_mae_axon_hm_multidata",
        ):
            with self.subTest(model=name):
                module = importlib.import_module(name)
                model = module.MaskedAutoencoderViT(
                    vox_size=32 if "cn_" in name else (32, 32, 32),
                    in_chans=2,
                    embed_dim=48,
                    depth=1,
                    num_heads=4,
                    decoder_embed_dim=32,
                    decoder_depth=1,
                    decoder_num_heads=4,
                ).eval()
                model.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    output = (
                        model(inputs, torch.zeros(1)) if "hm_multidata" in name else model(inputs)
                    )
                tensors = output.values() if isinstance(output, dict) else output
                for tensor in tensors:
                    self.assertEqual(tuple(tensor.shape[-3:]), (16, 16, 16))
                    self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
