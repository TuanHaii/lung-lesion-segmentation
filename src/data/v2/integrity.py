"""Data Integrity Validator for Pipeline V2.

Verifies:
1. Patient-level split integrity (zero patient leakage across train, val, test).
2. Dual-source distribution across splits.
3. Path validity and pairing non-identity (CT != Mask).
4. Label validity and slice counts.
5. Emits structured report separating Warnings from Critical Errors.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import csv


@dataclass
class IntegrityReport:
    total_patients: int = 0
    total_slices: int = 0
    coronacases_patients: int = 0
    radiopaedia_patients: int = 0
    train_patients: int = 0
    val_patients: int = 0
    test_patients: int = 0
    train_slices: int = 0
    val_slices: int = 0
    test_slices: int = 0
    lesion_slices: int = 0
    normal_slices: int = 0
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status_str = "PASSED" if self.passed else "FAILED"
        lines = [
            f"=== DATA INTEGRITY VALIDATION REPORT [{status_str}] ===",
            f"Total Patients: {self.total_patients} (Coronacases: {self.coronacases_patients}, Radiopaedia: {self.radiopaedia_patients})",
            f"Split Patients: Train={self.train_patients}, Val={self.val_patients}, Test={self.test_patients}",
            f"Total Slices:   {self.total_slices} (Train: {self.train_slices}, Val: {self.val_slices}, Test: {self.test_slices})",
            f"Lesion Distribution: {self.lesion_slices} lesion slices, {self.normal_slices} normal slices",
            f"Errors:   {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("\nErrors Detail:")
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")
        if self.warnings:
            lines.append("\nWarnings Detail:")
            for w in self.warnings:
                lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


def validate_dataset_integrity(
    manifest_path: str = "data/processed_v2/clean_slices_manifest_v2.csv",
    split_json_path: str = "data/processed/splits_patient_level.json",
    check_raw_files_exist: bool = False,
) -> IntegrityReport:
    """Run end-to-end data integrity validation on Manifest V2 and split definitions.

    Parameters
    ----------
    manifest_path : str
        Path to manifest V2 CSV.
    split_json_path : str
        Path to split JSON.
    check_raw_files_exist : bool
        If True, checks that every referenced raw file physically exists on disk.
        Set to False in environments where raw data is hosted remotely (e.g. Kaggle).

    Returns
    -------
    IntegrityReport
    """
    report = IntegrityReport()

    # 1. Validate Split JSON
    p_split = Path(split_json_path)
    if not p_split.exists():
        report.errors.append(f"Split JSON not found: {split_json_path}")
        report.passed = False
        return report

    with open(p_split, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    patient_splits = split_data.get("patient_splits", {})
    train_p = set(patient_splits.get("train", []))
    val_p = set(patient_splits.get("val", []))
    test_p = set(patient_splits.get("test", []))

    # Check zero patient leakage
    leak_train_val = train_p.intersection(val_p)
    leak_train_test = train_p.intersection(test_p)
    leak_val_test = val_p.intersection(test_p)

    if leak_train_val:
        report.errors.append(f"Patient leakage between Train and Val: {leak_train_val}")
    if leak_train_test:
        report.errors.append(f"Patient leakage between Train and Test: {leak_train_test}")
    if leak_val_test:
        report.errors.append(f"Patient leakage between Val and Test: {leak_val_test}")

    all_patients = train_p.union(val_p).union(test_p)
    report.total_patients = len(all_patients)
    report.train_patients = len(train_p)
    report.val_patients = len(val_p)
    report.test_patients = len(test_p)

    for p in all_patients:
        if "coronacases" in p.lower():
            report.coronacases_patients += 1
        elif "radiopaedia" in p.lower():
            report.radiopaedia_patients += 1
        else:
            report.warnings.append(f"Patient ID without standard source prefix: '{p}'")

    # 2. Validate Manifest Records
    p_manifest = Path(manifest_path)
    if not p_manifest.exists():
        report.errors.append(f"Manifest V2 not found: {manifest_path}")
        report.passed = False
        return report

    seen_sample_ids: Set[str] = set()
    slices_by_split: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "unassigned": 0}

    with open(p_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            report.total_slices += 1
            sid = row.get("sample_id", f"row_{i}")
            pid = row.get("patient_id", "")
            split = row.get("split", "unassigned")
            source = row.get("source", "")
            ct_path = row.get("source_volume_path", "")
            mask_path = row.get("source_mask_path", "")
            lesion_present = int(row.get("lesion_present", 0))

            # Duplicate check
            if sid in seen_sample_ids:
                report.errors.append(f"Duplicate sample_id found: '{sid}' at row {i}")
            seen_sample_ids.add(sid)

            # CT-mask identity check
            if ct_path and mask_path and ct_path.strip() == mask_path.strip():
                report.errors.append(
                    f"Sample '{sid}' has identical CT and Mask paths: '{ct_path}'"
                )

            # Source consistency
            if "coronacases" in pid.lower() and source != "coronacases":
                report.errors.append(f"Sample '{sid}' patient '{pid}' mismatch with source '{source}'")
            elif "radiopaedia" in pid.lower() and source != "radiopaedia":
                report.errors.append(f"Sample '{sid}' patient '{pid}' mismatch with source '{source}'")

            # Split consistency
            if pid not in all_patients:
                report.errors.append(f"Sample '{sid}' patient '{pid}' is not in split json!")
            else:
                expected_split = (
                    "train" if pid in train_p else ("val" if pid in val_p else "test")
                )
                if split != expected_split:
                    report.errors.append(
                        f"Sample '{sid}' split mismatch: row has '{split}', expected '{expected_split}'"
                    )

            slices_by_split[split] = slices_by_split.get(split, 0) + 1

            if lesion_present == 1:
                report.lesion_slices += 1
            else:
                report.normal_slices += 1

            # Optional physical file existence check
            if check_raw_files_exist:
                if not Path(ct_path).exists():
                    report.warnings.append(f"Raw CT not found on local disk: '{ct_path}'")
                if not Path(mask_path).exists():
                    report.warnings.append(f"Raw Mask not found on local disk: '{mask_path}'")

    report.train_slices = slices_by_split.get("train", 0)
    report.val_slices = slices_by_split.get("val", 0)
    report.test_slices = slices_by_split.get("test", 0)

    if slices_by_split.get("unassigned", 0) > 0:
        report.errors.append(f"Found {slices_by_split['unassigned']} unassigned slices in manifest!")

    if len(report.errors) > 0:
        report.passed = False

    return report


if __name__ == "__main__":
    rep = validate_dataset_integrity(
        manifest_path="data/processed/clean_slices_manifest.csv",
        split_json_path="data/processed/splits_patient_level.json",
    )
    print(rep.summary())
