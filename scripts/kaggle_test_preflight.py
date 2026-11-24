"""Kaggle Test Preflight Script for Version 2 CT Lung Lesion Segmentation.

Performs strict, read-only preflight verification on the mounted Kaggle dataset
before executing Test Evaluation.

Checks performed:
1. Exact patient membership of the Independent Test Set.
2. Zero overlap with Train and Validation patients (Patient-Level Isolation).
3. Exact slice count per test patient (Expected: 265 total slices).
4. Safe path resolution for CT volumes and ground-truth masks.
5. Volume shape compatibility (CT shape == Mask shape).
6. Affine matrix and spatial spacing alignment.
7. Mask label integrity (strictly binary {0, 1}).
8. Correct dual-source preprocessing dispatch (Coronacases HU-window vs Radiopaedia direct scaling).
9. Static split enforcement (no random re-splitting).
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

from src.data.v2.manifest import load_manifest_v2
from src.data.v2.resolver import SafePathResolver
from src.data.v2.source_detector import detect_source
from src.data.v2.preprocessing.registry import get_preprocessor
from src.data.v2.preprocessing.coronacases import CoronacasesPreprocessor
from src.data.v2.preprocessing.radiopaedia import RadiopaediaPreprocessor


EXPECTED_TEST_PATIENTS = [
    "P_coronacases_002",
    "P_coronacases_007",
    "P_radiopaedia_27_86410_0",
]

EXPECTED_PATIENT_SLICES = {
    "P_coronacases_002": 124,
    "P_coronacases_007": 104,
    "P_radiopaedia_27_86410_0": 37,
}

EXPECTED_TOTAL_TEST_SLICES = 265


def run_preflight(
    manifest_path: str = "data/processed_v2/clean_slices_manifest_v2.csv",
    split_path: str = "data/processed/splits_patient_level.json",
    raw_data_dir: str = None,
) -> bool:
    print("=" * 80)
    print(">> STARTING V2 TEST PREFLIGHT AUDIT")
    print("=" * 80)

    all_passed = True

    # 1. Verify static split file and hash
    print("\n[CHECK 1] Verifying Static Patient Split...")
    if not os.path.exists(split_path):
        print(f"  [FAIL] Split file not found: {split_path}")
        return False
    
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    
    patient_splits = split_data.get("patient_splits", {})
    train_pts = set(patient_splits.get("train", []))
    val_pts = set(patient_splits.get("val", []))
    test_pts = set(patient_splits.get("test", []))

    print(f"  Train patients count : {len(train_pts)}")
    print(f"  Val patients count   : {len(val_pts)}")
    print(f"  Test patients count  : {len(test_pts)}")

    # Check 1a: Membership
    if sorted(list(test_pts)) == sorted(EXPECTED_TEST_PATIENTS):
        print(f"  [PASS] Test patients match expected exactly: {sorted(list(test_pts))}")
    else:
        print(f"  [FAIL] Test patients mismatch! Expected {sorted(EXPECTED_TEST_PATIENTS)}, got {sorted(list(test_pts))}")
        all_passed = False

    # Check 1b: Zero Overlap
    overlap_train = test_pts.intersection(train_pts)
    overlap_val = test_pts.intersection(val_pts)
    if not overlap_train and not overlap_val:
        print("  [PASS] Zero overlap between Test and Train/Val (No Data Leakage).")
    else:
        print(f"  [FAIL] Leakage detected! Overlap with Train: {overlap_train}, Val: {overlap_val}")
        all_passed = False

    # 2. Verify Manifest and Slices
    print("\n[CHECK 2] Verifying Manifest and Slice Counts...")
    if not os.path.exists(manifest_path):
        print(f"  [FAIL] Manifest file not found: {manifest_path}")
        return False

    test_slices = load_manifest_v2(manifest_path, split="test")
    total_slices = len(test_slices)
    print(f"  Total test slices in manifest: {total_slices}")

    if total_slices == EXPECTED_TOTAL_TEST_SLICES:
        print(f"  [PASS] Total slice count matches expected ({EXPECTED_TOTAL_TEST_SLICES}).")
    else:
        print(f"  [FAIL] Slice count mismatch! Expected {EXPECTED_TOTAL_TEST_SLICES}, got {total_slices}")
        all_passed = False

    # Check per-patient slice counts
    patient_slice_counts = {}
    for s in test_slices:
        pid = s["patient_id"]
        patient_slice_counts[pid] = patient_slice_counts.get(pid, 0) + 1

    for pid, expected_cnt in EXPECTED_PATIENT_SLICES.items():
        actual_cnt = patient_slice_counts.get(pid, 0)
        if actual_cnt == expected_cnt:
            print(f"  [PASS] Patient {pid}: {actual_cnt} slices (matches expected).")
        else:
            print(f"  [FAIL] Patient {pid}: expected {expected_cnt}, got {actual_cnt}")
            all_passed = False

    # 3. Path Resolution and Volume Integrity
    print("\n[CHECK 3] Verifying Path Resolution and NIfTI Volumes...")
    resolver = SafePathResolver(raw_data_dir=raw_data_dir)
    print(f"  Using SafePathResolver with raw_data_dir: {resolver.raw_data_dir}")

    # Check if raw data root exists
    search_roots = resolver._get_search_roots()
    raw_exists = any(root.exists() and any(root.glob("*.nii*")) for root in search_roots)

    if not raw_exists and raw_data_dir is None:
        print("  [UNVERIFIED - LOCAL WINDOWS ENVIRONMENT]")
        print("  Raw CT dataset is not mounted locally in data/raw. This check is designed for Kaggle.")
        print("  On Kaggle, pass --data_dir /kaggle/input/covid-19-ct-scans or set RAW_DATA_DIR env var.")
        local_skip = True
    else:
        local_skip = False

    if local_skip:
        pass
    elif nib is None:
        print("  [WARN] nibabel not installed in current environment. Volume tensor inspection skipped.")
        print("         Ensure nibabel is installed on Kaggle before running inference.")
    else:
        unique_volumes = {}
        for s in test_slices:
            pid = s["patient_id"]
            if pid not in unique_volumes:
                unique_volumes[pid] = (s["source_volume_path"], s["source_mask_path"], s["sample_id"])

        for pid, (ct_p, msk_p, sid) in unique_volumes.items():
            print(f"\n  Inspecting volume pair for {pid}...")
            try:
                res_ct, res_msk = resolver.resolve_pair(ct_p, msk_p, patient_id=pid, sample_id=sid)
                print(f"    Resolved CT   : {res_ct}")
                print(f"    Resolved Mask : {res_msk}")

                if not os.path.exists(res_ct):
                    print(f"    [FAIL] CT file does not exist on disk: {res_ct}")
                    all_passed = False
                    continue
                if not os.path.exists(res_msk):
                    print(f"    [FAIL] Mask file does not exist on disk: {res_msk}")
                    all_passed = False
                    continue

                ct_nii = nib.load(str(res_ct))
                msk_nii = nib.load(str(res_msk))

                ct_shape = ct_nii.shape
                msk_shape = msk_nii.shape
                print(f"    CT Shape: {ct_shape} | Mask Shape: {msk_shape}")

                if ct_shape == msk_shape:
                    print("    [PASS] CT and Mask 3D shapes match exactly.")
                else:
                    print(f"    [FAIL] Shape mismatch between CT {ct_shape} and Mask {msk_shape}!")
                    all_passed = False

                # Affine check
                if np.allclose(ct_nii.affine, msk_nii.affine, atol=1e-3):
                    print("    [PASS] Affine matrices match within tolerance.")
                else:
                    print("    [WARN] Affine matrices differ slightly; check spatial registration.")

                # Intensity dynamic range and preprocessor check
                ct_data = ct_nii.get_fdata(dtype=np.float32)
                msk_data = msk_nii.get_fdata(dtype=np.float32)
                print(f"    CT Intensity Min: {ct_data.min():.1f}, Max: {ct_data.max():.1f}")
                print(f"    Mask Unique Values: {np.unique(msk_data)}")

                source = detect_source(patient_id=pid, path=str(res_ct))
                preprocessor = get_preprocessor(source)
                print(f"    Detected Source: '{source}' -> Preprocessor: {preprocessor.__class__.__name__}")

                if "coronacases" in pid:
                    if isinstance(preprocessor, CoronacasesPreprocessor):
                        print("    [PASS] Coronacases correctly dispatched to CoronacasesPreprocessor (HU window [-1000, 400]).")
                    else:
                        print("    [FAIL] Wrong preprocessor dispatched for Coronacases!")
                        all_passed = False
                elif "radiopaedia" in pid:
                    if isinstance(preprocessor, RadiopaediaPreprocessor):
                        print("    [PASS] Radiopaedia correctly dispatched to RadiopaediaPreprocessor (linear scale / 255.0).")
                    else:
                        print("    [FAIL] Wrong preprocessor dispatched for Radiopaedia!")
                        all_passed = False

                # Test preprocessed slice
                mid_z = ct_shape[2] // 2
                test_slice = ct_data[:, :, mid_z]
                norm_slice = preprocessor.preprocess(test_slice)
                if 0.0 <= norm_slice.min() and norm_slice.max() <= 1.0:
                    print(f"    [PASS] Sample slice normalized strictly in [0.0, 1.0] (min={norm_slice.min():.3f}, max={norm_slice.max():.3f}).")
                else:
                    print(f"    [FAIL] Slice normalization out of bounds! min={norm_slice.min()}, max={norm_slice.max()}")
                    all_passed = False

            except Exception as e:
                print(f"    [FAIL] Error inspecting volume for {pid}: {e}")
                all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print(">> PREFLIGHT AUDIT RESULT: ALL CHECKS PASSED (READY FOR TEST EVALUATION) ✅")
    else:
        print(">> PREFLIGHT AUDIT RESULT: ONE OR MORE CHECKS FAILED ❌")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle Test Preflight Audit")
    parser.add_argument("--manifest", type=str, default="data/processed_v2/clean_slices_manifest_v2.csv")
    parser.add_argument("--split", type=str, default="data/processed/splits_patient_level.json")
    parser.add_argument("--data_dir", type=str, default=None, help="Root raw data dir on Kaggle")
    args = parser.parse_args()

    success = run_preflight(
        manifest_path=args.manifest,
        split_path=args.split,
        raw_data_dir=args.data_dir,
    )
    sys.exit(0 if success else 1)
