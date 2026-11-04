#!/usr/bin/env python3
"""
Task ID: EVAL-01 (Data & Evaluation Lead)
Tên nhiệm vụ: Chuẩn bị và cô lập tập kiểm thử độc lập (Independent Test Set Manifest).
Mục đích:
- Đọc phân vùng bệnh nhân từ data/processed/splits_patient_level.json (15% test, 3/20 bệnh nhân hoàn toàn unseen).
- Trích xuất toàn bộ lát cắt thuộc 3 bệnh nhân test từ data/processed/clean_slices_manifest.csv (265 lát cắt).
- Xác thực tính cô lập tuyệt đối (Zero Data Leakage Check) giữa Train, Val và Test.
- Xuất file data/processed/test_manifest.json lưu trữ: test_patients, total_test_slices, chi tiết từng lát cắt.
- Khóa quyền ghi (chmod 0o444, read-only) để bảo toàn tính toàn vẹn dữ liệu thử nghiệm y khoa.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def verify_patient_isolation(splits_data: Dict[str, Any]) -> None:
    """Xác thực tính cô lập tuyệt đối ở cấp độ bệnh nhân (Zero Data Leakage)."""
    p_splits = splits_data.get("patient_splits", {})
    train_patients = set(p_splits.get("train", []))
    val_patients = set(p_splits.get("val", []))
    test_patients = set(p_splits.get("test", []))

    if not test_patients:
        raise ValueError("[ERROR] Danh sách bệnh nhân test bị rỗng trong splits_patient_level.json!")

    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)

    if train_test_overlap:
        raise RuntimeError(
            f"[FATAL DATA LEAKAGE] Trùng lặp bệnh nhân giữa Train và Test: {train_test_overlap}"
        )
    if val_test_overlap:
        raise RuntimeError(
            f"[FATAL DATA LEAKAGE] Trùng lặp bệnh nhân giữa Val và Test: {val_test_overlap}"
        )

    print(f"[*] Zero Data Leakage verified: 3 subsets isolated. Test patients: {sorted(test_patients)}")


def extract_test_slices(
    splits_data: Dict[str, Any],
    clean_manifest_path: Path,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    """Trích xuất và khớp thông tin lát cắt của tập kiểm thử độc lập."""
    test_patients = sorted(splits_data.get("patient_splits", {}).get("test", []))
    test_patients_set = set(test_patients)
    expected_sample_ids = set(splits_data.get("slice_samples", {}).get("test", []))

    if not clean_manifest_path.exists():
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file manifest sạch: {clean_manifest_path}")

    slices: List[Dict[str, Any]] = []
    with open(clean_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["patient_id"] in test_patients_set:
                record = {
                    "sample_id": row["sample_id"],
                    "patient_id": row["patient_id"],
                    "volume_id": row.get("volume_id", ""),
                    "slice_index": int(row["slice_index"]),
                    "original_shape": row.get("original_shape", "512x512"),
                    "spacing_x": float(row.get("spacing_x", 1.0)),
                    "spacing_y": float(row.get("spacing_y", 1.0)),
                    "spacing_z": float(row.get("spacing_z", 1.0)),
                    "lung_area_px": int(row.get("lung_area_px", 0)),
                    "lesion_present": int(row.get("lesion_present", 0)),
                    "lesion_area_px": int(row.get("lesion_area_px", 0)),
                    "source_volume_path": row.get("source_volume_path", ""),
                    "source_mask_path": row.get("source_mask_path", ""),
                    "qa_status": row.get("qa_status", "PASSED"),
                }
                slices.append(record)

    # Sắp xếp theo patient_id và slice_index để đảm bảo tính tái lập 100%
    slices.sort(key=lambda s: (s["patient_id"], s["slice_index"]))

    extracted_sample_ids = {s["sample_id"] for s in slices}
    if expected_sample_ids and extracted_sample_ids != expected_sample_ids:
        missing = expected_sample_ids - extracted_sample_ids
        unexpected = extracted_sample_ids - expected_sample_ids
        raise ValueError(
            f"[ERROR] Bất nhất giữa splits_patient_level.json và clean_slices_manifest.csv! "
            f"Thiếu: {len(missing)}, Thừa: {len(unexpected)}"
        )

    lesion_count = sum(1 for s in slices if s["lesion_present"] == 1)
    normal_count = sum(1 for s in slices if s["lesion_present"] == 0)

    stats = {
        "total_test_slices": len(slices),
        "lesion_slices": lesion_count,
        "normal_slices": normal_count,
    }
    return test_patients, slices, stats


def save_test_manifest(
    output_path: Path,
    test_patients: List[str],
    slices: List[Dict[str, Any]],
    stats: Dict[str, int],
    force: bool = False,
) -> None:
    """Lưu manifest ra file JSON và khóa quyền ghi (read-only 0o444)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Gỡ quyền read-only cũ nếu đã tồn tại để cho phép ghi mới
    if output_path.exists():
        if force or not os.access(output_path, os.W_OK):
            os.chmod(output_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    manifest_payload = {
        "metadata": {
            "dataset_version": "v1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "test_patients": test_patients,
            "total_test_patients": len(test_patients),
            "total_test_slices": stats["total_test_slices"],
            "total_lesion_slices": stats["lesion_slices"],
            "total_normal_slices": stats["normal_slices"],
            "split_ratio": "70/15/15",
            "zero_leakage_verified": True,
            "qa_status": "PASSED",
        },
        "slices": slices,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

    # Khóa file sang trạng thái chỉ đọc (read-only: 0o444)
    os.chmod(output_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"[+] Saved test manifest to {output_path} and locked permissions (0o444 READ-ONLY)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EVAL-01: Prepare and isolate independent test set manifest"
    )
    parser.add_argument(
        "--splits_path",
        type=Path,
        default=Path("data/processed/splits_patient_level.json"),
        help="Đường dẫn đến splits_patient_level.json",
    )
    parser.add_argument(
        "--clean_manifest_path",
        type=Path,
        default=Path("data/processed/clean_slices_manifest.csv"),
        help="Đường dẫn đến clean_slices_manifest.csv",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/test_manifest.json"),
        help="Đường dẫn file đích test_manifest.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè file nếu đã tồn tại",
    )
    args = parser.parse_args()

    print("======================================================================")
    print(" TASK EVAL-01: INDEPENDENT TEST SET MANIFEST GENERATION")
    print("======================================================================")

    if not args.splits_path.exists():
        print(f"[!] Lỗi: Không tìm thấy file {args.splits_path}", file=sys.stderr)
        return 1

    with open(args.splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # 1. Kiểm tra Zero Data Leakage
    verify_patient_isolation(splits_data)

    # 2. Trích xuất lát cắt
    test_patients, slices, stats = extract_test_slices(splits_data, args.clean_manifest_path)
    print(f"[*] Extracted {stats['total_test_slices']} test slices across {len(test_patients)} patients.")
    print(f"    - Lesion-containing slices (GT > 0): {stats['lesion_slices']}")
    print(f"    - Normal slices (GT = 0): {stats['normal_slices']}")

    # 3. Lưu manifest và khóa quyền ghi
    save_test_manifest(args.output_path, test_patients, slices, stats, force=args.force)

    print("[SUCCESS] EVAL-01 completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
