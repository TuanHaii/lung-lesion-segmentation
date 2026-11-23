"""Comprehensive Test Suite for Pipeline V2.

Verifies:
1. SourceDetector (strict identification, error handling)
2. SafePathResolver (type safety, distinct pairing, no unsafe fallback)
3. Dual-source Preprocessing (Coronacases HU vs Radiopaedia direct scaling)
4. Spatial resizing and binary mask thresholding
5. Data Integrity (Manifest V2, zero patient leakage)
6. DataLoader V2 (batch shapes, dtypes, metadata)
7. Model forward shape compatibility (UNet, AttentionUNet, SegNet) - Zero optimizer step!
8. Independent Evaluator V2 (Overall micro, macro, empty-mask handling, manual math match)
"""

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from src.data.v2.source_detector import detect_source, SourceIdentificationError
from src.data.v2.resolver import SafePathResolver, PathResolverError
from src.data.v2.preprocessing.coronacases import CoronacasesPreprocessor
from src.data.v2.preprocessing.radiopaedia import RadiopaediaPreprocessor
from src.data.v2.preprocessing.common import resize_ct_pair_v2, binarize_mask_v2
from src.data.v2.manifest import create_manifest_v2, load_manifest_v2
from src.data.v2.integrity import validate_dataset_integrity
from src.data.v2.dataset import LungCTDatasetV2, get_dataloader_v2
from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.models.segnet import SegNet
from src.evaluation.v2_evaluator import EvaluatorV2, compute_slice_confusion
from src.utils.metrics import compute_sample_metrics


class TestSourceDetector(unittest.TestCase):
    def test_coronacases_detection(self):
        self.assertEqual(detect_source(patient_id="P_coronacases_001"), "coronacases")
        self.assertEqual(detect_source(sample_id="P_coronacases_002_z0010"), "coronacases")
        self.assertEqual(detect_source(path="data/raw/ct_volumes/coronacases_003.nii.gz"), "coronacases")
        self.assertEqual(detect_source(metadata={"source": "coronacases"}), "coronacases")

    def test_radiopaedia_detection(self):
        self.assertEqual(detect_source(patient_id="P_radiopaedia_10_85902_1"), "radiopaedia")
        self.assertEqual(detect_source(sample_id="P_radiopaedia_27_z0005"), "radiopaedia")
        self.assertEqual(detect_source(path="data/raw/ct_volumes/radiopaedia_40.nii.gz"), "radiopaedia")
        self.assertEqual(detect_source(metadata={"source": "radiopaedia"}), "radiopaedia")

    def test_unknown_source_raises(self):
        with self.assertRaises(SourceIdentificationError):
            detect_source(sample_id="patient_xyz_123", patient_id="unknown_p", path="data/raw/scan.nii")


class TestSafePathResolver(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

        # Create realistic directory structure
        self.ct_dir = self.root / "ct_volumes"
        self.mask_dir = self.root / "ground_truth_masks"
        self.ct_dir.mkdir(parents=True)
        self.mask_dir.mkdir(parents=True)

        # Create valid non-empty files
        self.valid_ct = self.ct_dir / "coronacases_002.nii.gz"
        self.valid_mask = self.mask_dir / "coronacases_002.nii.gz"
        self.valid_ct.write_bytes(b"dummy ct data")
        self.valid_mask.write_bytes(b"dummy mask data")

        self.resolver = SafePathResolver(raw_data_dir=str(self.root))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_resolve_valid_pair(self):
        ct_p, mask_p = self.resolver.resolve_pair(
            ct_path="coronacases_002.nii.gz",
            mask_path="coronacases_002.nii.gz",
        )
        self.assertEqual(ct_p.resolve(), self.valid_ct.resolve())
        self.assertEqual(mask_p.resolve(), self.valid_mask.resolve())
        self.assertNotEqual(ct_p.resolve(), mask_p.resolve())

    def test_prevent_ct_mask_identity(self):
        # If someone asks to resolve the exact same CT file for both
        with self.assertRaises(PathResolverError):
            self.resolver.resolve_pair(
                ct_path=str(self.valid_ct),
                mask_path=str(self.valid_ct),
            )

    def test_missing_file_raises_filenotfound(self):
        # Non-existent file must raise FileNotFoundError, NOT fallback to random file
        with self.assertRaises(FileNotFoundError):
            self.resolver.resolve_file("non_existent_scan_999.nii.gz", file_type="ct")

    def test_type_isolation_enforcement(self):
        # A file with 'mask' in its name cannot be resolved as a CT file
        bad_ct = self.ct_dir / "infection_mask_001.nii.gz"
        bad_ct.write_bytes(b"mask data")
        with self.assertRaises(PathResolverError):
            self.resolver.resolve_file(str(bad_ct), file_type="ct")


class TestPreprocessingProfiles(unittest.TestCase):
    def test_coronacases_hu_windowing(self):
        prep = CoronacasesPreprocessor(min_hu=-1000.0, max_hu=400.0)

        # Values below -1000 should be clipped to 0.0
        # -1000 -> 0.0
        # 400 -> 1.0
        # Values above 400 should be clipped to 1.0
        # Midpoint: -300 -> (-300 - (-1000)) / 1400 = 700 / 1400 = 0.5
        raw_slice = np.array([
            [-1500.0, -1000.0],
            [-300.0, 400.0],
            [1000.0, 0.0],
        ], dtype=np.float32)

        out = prep.preprocess(raw_slice)

        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(out[0, 0], 0.0, places=5)
        self.assertAlmostEqual(out[0, 1], 0.0, places=5)
        self.assertAlmostEqual(out[1, 0], 0.5, places=5)
        self.assertAlmostEqual(out[1, 1], 1.0, places=5)
        self.assertAlmostEqual(out[2, 0], 1.0, places=5)
        self.assertAlmostEqual(out[2, 1], 1000.0 / 1400.0, places=5)
        self.assertTrue(out.min() >= 0.0 and out.max() <= 1.0)

    def test_radiopaedia_direct_scaling(self):
        prep = RadiopaediaPreprocessor(max_value=255.0)

        # Radiopaedia uint8 [0, 255]
        # 0 -> 0.0
        # 255 -> 1.0
        # 127.5 -> 0.5
        raw_slice = np.array([
            [0, 127.5],
            [255, 51],
        ], dtype=np.float32)

        out = prep.preprocess(raw_slice)

        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(out[0, 0], 0.0, places=5)
        self.assertAlmostEqual(out[0, 1], 0.5, places=5)
        self.assertAlmostEqual(out[1, 0], 1.0, places=5)
        self.assertAlmostEqual(out[1, 1], 51.0 / 255.0, places=5)
        self.assertTrue(out.min() >= 0.0 and out.max() <= 1.0)

    def test_nan_inf_handling(self):
        prep_c = CoronacasesPreprocessor()
        bad_c = np.array([[np.nan, np.inf], [-np.inf, -1000.0]], dtype=np.float32)
        out_c = prep_c.preprocess(bad_c)
        self.assertFalse(np.isnan(out_c).any())
        self.assertFalse(np.isinf(out_c).any())
        self.assertTrue(out_c.min() >= 0.0 and out_c.max() <= 1.0)

        prep_r = RadiopaediaPreprocessor()
        bad_r = np.array([[np.nan, np.inf], [-np.inf, 128.0]], dtype=np.float32)
        out_r = prep_r.preprocess(bad_r)
        self.assertFalse(np.isnan(out_r).any())
        self.assertFalse(np.isinf(out_r).any())
        self.assertTrue(out_r.min() >= 0.0 and out_r.max() <= 1.0)

    def test_spatial_resize_and_binarize(self):
        img = np.random.uniform(0.0, 1.0, size=(512, 512)).astype(np.float32)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 1
        mask[300:400, 300:400] = 2  # other class

        # Resize to 256x256
        res_img, res_mask = resize_ct_pair_v2(img, mask, target_size=(256, 256))
        self.assertEqual(res_img.shape, (256, 256))
        self.assertEqual(res_mask.shape, (256, 256))

        # Binary extraction for lesion_label=1
        bin_mask = binarize_mask_v2(res_mask, lesion_label=1)
        self.assertEqual(bin_mask.shape, (256, 256))
        self.assertEqual(bin_mask.dtype, np.float32)
        self.assertTrue(set(np.unique(bin_mask)).issubset({0.0, 1.0}))
        self.assertTrue(np.sum(bin_mask == 1.0) > 0)


class TestDataIntegrityAndManifest(unittest.TestCase):
    def test_manifest_v2_generation_and_integrity(self):
        records = create_manifest_v2(
            v1_manifest_path="data/processed/clean_slices_manifest.csv",
            split_json_path="data/processed/splits_patient_level.json",
            output_manifest_path="data/processed_v2/clean_slices_manifest_v2.csv",
        )
        self.assertEqual(len(records), 1750)

        report = validate_dataset_integrity(
            manifest_path="data/processed_v2/clean_slices_manifest_v2.csv",
            split_json_path="data/processed/splits_patient_level.json",
        )
        self.assertTrue(report.passed)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(report.total_patients, 20)
        self.assertEqual(report.train_patients, 14)
        self.assertEqual(report.val_patients, 3)
        self.assertEqual(report.test_patients, 3)
        self.assertEqual(report.total_slices, 1750)
        self.assertEqual(report.train_slices, 1277)
        self.assertEqual(report.val_slices, 208)
        self.assertEqual(report.test_slices, 265)


class TestModelForwardCompatibility(unittest.TestCase):
    """Test model forward pass with dummy tensor. Zero optimizer step!"""

    def test_unet_forward_shape(self):
        model = UNet(in_channels=1, out_channels=1, features=[16, 32, 64, 128])
        model.eval()
        dummy_input = torch.randn(2, 1, 256, 256, dtype=torch.float32)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (2, 1, 256, 256))

    def test_attention_unet_forward_shape(self):
        model = AttentionUNet(in_channels=1, out_channels=1, features=[16, 32, 64, 128])
        model.eval()
        dummy_input = torch.randn(2, 1, 256, 256, dtype=torch.float32)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (2, 1, 256, 256))

    def test_segnet_forward_shape(self):
        model = SegNet(in_channels=1, num_classes=1, init_features=16)
        model.eval()
        dummy_input = torch.randn(2, 1, 256, 256, dtype=torch.float32)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (2, 1, 256, 256))


class TestEvaluatorV2Math(unittest.TestCase):
    def test_confusion_matrix_manual_math(self):
        # 4 pixels:
        # Pred: [1, 1, 0, 0]
        # GT:   [1, 0, 1, 0]
        # TP=1, FP=1, FN=1, TN=1
        pred = np.array([1, 1, 0, 0], dtype=np.uint8)
        gt = np.array([1, 0, 1, 0], dtype=np.uint8)
        conf = compute_slice_confusion(pred, gt)

        self.assertEqual(conf["tp"], 1)
        self.assertEqual(conf["fp"], 1)
        self.assertEqual(conf["fn"], 1)
        self.assertEqual(conf["tn"], 1)

        # Expected Dice: (2*1) / (2*1 + 1 + 1) = 2/4 = 0.5
        self.assertAlmostEqual(conf["dice"], 0.5, places=4)
        # Expected IoU: 1 / (1 + 1 + 1) = 1/3 = 0.333333
        self.assertAlmostEqual(conf["iou"], 1.0 / 3.0, places=4)
        # Expected Precision: 1 / (1 + 1) = 0.5
        self.assertAlmostEqual(conf["precision"], 0.5, places=4)
        # Expected Recall: 1 / (1 + 1) = 0.5
        self.assertAlmostEqual(conf["recall"], 0.5, places=4)

    def test_empty_mask_handling(self):
        # Both empty: GT=0, Pred=0 -> Dice=1.0, IoU=1.0
        empty_pred = np.zeros((10, 10), dtype=np.uint8)
        empty_gt = np.zeros((10, 10), dtype=np.uint8)
        conf_both_empty = compute_slice_confusion(empty_pred, empty_gt)
        self.assertEqual(conf_both_empty["dice"], 1.0)
        self.assertEqual(conf_both_empty["iou"], 1.0)
        self.assertEqual(conf_both_empty["precision"], 1.0)
        self.assertEqual(conf_both_empty["recall"], 1.0)

        # GT empty, Pred non-empty (False Positive): Dice=0.0
        fp_pred = np.ones((10, 10), dtype=np.uint8)
        conf_fp = compute_slice_confusion(fp_pred, empty_gt)
        self.assertEqual(conf_fp["dice"], 0.0)
        self.assertEqual(conf_fp["iou"], 0.0)
        self.assertEqual(conf_fp["precision"], 0.0)

    def test_macro_sample_metrics_torch(self):
        # Batch of 2 samples:
        # Sample 0: perfect match (Dice=1.0)
        # Sample 1: completely disjoint (Dice=0.0)
        # Batch macro dice should be exactly (1.0 + 0.0) / 2 = 0.5
        preds = torch.tensor([
            [[[10.0, 10.0], [-10.0, -10.0]]],  # sample 0
            [[[10.0, 10.0], [-10.0, -10.0]]],  # sample 1
        ], dtype=torch.float32)

        gts = torch.tensor([
            [[[1.0, 1.0], [0.0, 0.0]]],  # sample 0 (matches pred)
            [[[0.0, 0.0], [1.0, 1.0]]],  # sample 1 (disjoint from pred)
        ], dtype=torch.float32)

        m = compute_sample_metrics(preds, gts, threshold=0.5)
        self.assertAlmostEqual(m["dice"], 0.5, places=4)

    def test_multi_level_evaluation_report(self):
        evaluator = EvaluatorV2(threshold=0.5)

        # 2 coronacases slices + 2 radiopaedia slices
        slice_items = [
            {
                "sample_id": "P_coronacases_001_z0010",
                "patient_id": "P_coronacases_001",
                "source": "coronacases",
                "slice_index": 10,
                "lesion_present": 1,
                "pred": np.array([[1, 0], [0, 0]]),
                "gt": np.array([[1, 0], [0, 0]]),
            },
            {
                "sample_id": "P_coronacases_001_z0011",
                "patient_id": "P_coronacases_001",
                "source": "coronacases",
                "slice_index": 11,
                "lesion_present": 0,
                "pred": np.array([[0, 0], [0, 0]]),
                "gt": np.array([[0, 0], [0, 0]]),
            },
            {
                "sample_id": "P_radiopaedia_10_z0001",
                "patient_id": "P_radiopaedia_10",
                "source": "radiopaedia",
                "slice_index": 1,
                "lesion_present": 1,
                "pred": np.array([[1, 1], [0, 0]]),
                "gt": np.array([[1, 0], [0, 0]]),
            },
            {
                "sample_id": "P_radiopaedia_10_z0002",
                "patient_id": "P_radiopaedia_10",
                "source": "radiopaedia",
                "slice_index": 2,
                "lesion_present": 0,
                "pred": np.array([[0, 0], [0, 0]]),
                "gt": np.array([[0, 0], [0, 0]]),
            },
        ]

        report = evaluator.evaluate_slices(slice_items)
        self.assertIn("overall", report)
        self.assertIn("by_source", report)
        self.assertIn("by_patient", report)
        self.assertIn("coronacases", report["by_source"])
        self.assertIn("radiopaedia", report["by_source"])
        self.assertIn("P_coronacases_001", report["by_patient"])
        self.assertIn("P_radiopaedia_10", report["by_patient"])
        self.assertEqual(report["overall"]["total_slices"], 4)


class TestDataLoaderV2(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

        # Create dummy NIfTI volumes using nibabel
        import nibabel as nib
        ct_data = np.full((64, 64, 5), fill_value=-500.0, dtype=np.float32)
        mask_data = np.zeros((64, 64, 5), dtype=np.uint8)
        mask_data[20:40, 20:40, :] = 1

        ct_nii = nib.Nifti1Image(ct_data, np.eye(4))
        mask_nii = nib.Nifti1Image(mask_data, np.eye(4))

        ct_dir = self.root / "ct_volumes"
        mask_dir = self.root / "ground_truth_masks"
        ct_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        self.ct_file = ct_dir / "coronacases_999.nii.gz"
        self.mask_file = mask_dir / "coronacases_999.nii.gz"
        nib.save(ct_nii, str(self.ct_file))
        nib.save(mask_nii, str(self.mask_file))

        # Build mock sample records
        self.mock_samples = [
            {
                "sample_id": f"P_coronacases_999_z000{z}",
                "patient_id": "P_coronacases_999",
                "source": "coronacases",
                "slice_index": z,
                "lesion_present": 1,
                "source_volume_path": str(self.ct_file),
                "source_mask_path": str(self.mask_file),
                "split": "train" if z < 3 else ("val" if z == 3 else "test"),
            }
            for z in range(5)
        ]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_dataloader_train_val_test_loading(self):
        for split in ["train", "val", "test"]:
            loader = get_dataloader_v2(
                split=split,
                batch_size=2,
                target_size=(256, 256),
                lesion_label=1,
                raw_data_dir=str(self.root),
                samples_override=self.mock_samples,
            )
            for images, masks, metadata in loader:
                self.assertEqual(images.shape[1:], (1, 256, 256))
                self.assertEqual(masks.shape[1:], (1, 256, 256))
                self.assertEqual(images.dtype, torch.float32)
                self.assertEqual(masks.dtype, torch.float32)
                self.assertTrue(images.min() >= 0.0 and images.max() <= 1.0)
                self.assertTrue(set(torch.unique(masks).tolist()).issubset({0.0, 1.0}))
                self.assertEqual(metadata["split"][0], split)


if __name__ == "__main__":
    unittest.main()
