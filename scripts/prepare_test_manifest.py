#!/usr/bin/env python3
"""
Task ID: EVAL-01 (Tuần 5 - Phase: Evaluation & Analysis)
Tên nhiệm vụ: Chuẩn bị và cô lập tập kiểm thử độc lập (Independent Test Set).
Mục đích:
- Đọc phân vùng bệnh nhân từ data/processed/splits_patient_level.json (15% test, 3/20 ca hoàn toàn unseen).
- Trích xuất danh mục lát cắt tương ứng từ data/processed/clean_slices_manifest.csv (265 lát cắt).
- Xác thực tính cô lập tuyệt đối (Zero Data Leakage) ở cả cấp độ bệnh nhân và lát cắt.
- Đối chiếu tính nhất quán với bộ dữ liệu đã niêm phong (data/processed/dataset_v1.0_freeze_manifest.json).
- Xuất dữ liệu ra file data/processed/test_manifest.json kèm metadata đầy đủ.
- Thiết lập quyền chỉ đọc (read-only, 0o444) để khóa file, ngăn chặn mọi sửa đổi ngoài ý muốn.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Dict, List, Set, Tuple


def compute_sha256(file_path: Path) -> str:
    """Tính toán mã băm SHA-256 phục vụ xác thực toàn vẹn dữ liệu."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_patient_and_slice_isolation(
    splits_data: Dict[str, Any],
    clean_manifest_rows: List[Dict[str, str]],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Xác thực nghiêm ngặt tính cô lập của tập kiểm thử độc lập (Zero Leakage).

    Parameters
    ----------
    splits_data : Dict[str, Any]
        Dữ liệu đọc từ splits_patient_level.json.
    clean_manifest_rows : List[Dict[str, str]]
        Danh sách các dòng đọc từ clean_slices_manifest.csv.

    Returns
    -------
    Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]
        Danh sách test_patients, danh sách test_slices_details, và summary stats.
    """
    patient_splits = splits_data.get("patient_splits", {})
    train_patients: Set[str] = set(patient_splits.get("train", []))
    val_patients: Set[str] = set(patient_splits.get("val", []))
    test_patients: Set[str] = set(patient_splits.get("test", []))

    # 1. Kiểm tra tính giao thoa ở cấp độ bệnh nhân
    assert len(test_patients) > 0, "Lỗi: Danh sách test_patients trong splits bị rỗng!"
    assert train_patients.isdisjoint(test_patients), (
        f"RÒ RỈ DỮ LIỆU: Bệnh nhân trùng giữa Train và Test: {train_patients.intersection(test_patients)}"
    )
    assert val_patients.isdisjoint(test_patients), (
        f"RÒ RỈ DỮ LIỆU: Bệnh nhân trùng giữa Val và Test: {val_patients.intersection(test_patients)}"
    )
    assert train_patients.isdisjoint(val_patients), (
        f"RÒ RỈ DỮ LIỆU: Bệnh nhân trùng giữa Train và Val: {train_patients.intersection(val_patients)}"
    )

    # 2. Kiểm tra tính giao thoa ở cấp độ mẫu lát cắt (slice_samples)
    slice_samples = splits_data.get("slice_samples", {})
    train_slice_ids: Set[str] = set(slice_samples.get("train", []))
    val_slice_ids: Set[str] = set(slice_samples.get("val", []))
    test_slice_ids: Set[str] = set(slice_samples.get("test", []))

    assert len(test_slice_ids) > 0, "Lỗi: Danh sách slice test trong splits bị rỗng!"
    assert train_slice_ids.isdisjoint(test_slice_ids), (
        f"RÒ RỈ DỮ LIỆU: Lát cắt trùng giữa Train và Test: {train_slice_ids.intersection(test_slice_ids)}"
    )
    assert val_slice_ids.isdisjoint(test_slice_ids), (
        f"RÒ RỈ DỮ LIỆU: Lát cắt trùng giữa Val và Test: {val_slice_ids.intersection(test_slice_ids)}"
    )
    assert train_slice_ids.isdisjoint(val_slice_ids), (
        f"RÒ RỈ DỮ LIỆU: Lát cắt trùng giữa Train và Val: {train_slice_ids.intersection(val_slice_ids)}"
    )

    # 3. Lọc và ánh xạ các lát cắt thuộc tập Test từ clean_slices_manifest.csv
    test_slices: List[Dict[str, Any]] = []
    test_patients_found: Set[str] = set()
    lesion_slices_count = 0
    normal_slices_count = 0
    patient_slice_counts: Dict[str, int] = {}
    patient_lesion_counts: Dict[str, Dict[str, int]] = {}

    for row in clean_manifest_rows:
        sample_id = row["sample_id"]
        pid = row["patient_id"]

        # Nếu lát cắt thuộc tập test
        if sample_id in test_slice_ids or pid in test_patients:
            # Xác thực chéo: bệnh nhân phải thuộc tập test và sample_id phải nằm trong test_slice_ids
            assert pid in test_patients, (
                f"Lỗi bất nhất: sample {sample_id} có patient_id {pid} không thuộc test_patients!"
            )
            assert sample_id in test_slice_ids, (
                f"Lỗi bất nhất: patient {pid} có sample_id {sample_id} không nằm trong test_slice_ids!"
            )

            test_patients_found.add(pid)
            lesion_present = int(row["lesion_present"])
            if lesion_present == 1:
                lesion_slices_count += 1
            else:
                normal_slices_count += 1

            patient_slice_counts[pid] = patient_slice_counts.get(pid, 0) + 1
            if pid not in patient_lesion_counts:
                patient_lesion_counts[pid] = {"lesion_slices": 0, "normal_slices": 0, "total": 0}
            if lesion_present == 1:
                patient_lesion_counts[pid]["lesion_slices"] += 1
            else:
                patient_lesion_counts[pid]["normal_slices"] += 1
            patient_lesion_counts[pid]["total"] += 1

            # Chuẩn hóa kiểu dữ liệu cho từng trường
            slice_record: Dict[str, Any] = {
                "sample_id": sample_id,
                "patient_id": pid,
                "volume_id": row.get("volume_id", ""),
                "slice_index": int(row["slice_index"]),
                "original_shape": row.get("original_shape", "512x512"),
                "spacing_x": float(row["spacing_x"]) if row.get("spacing_x") else None,
                "spacing_y": float(row["spacing_y"]) if row.get("spacing_y") else None,
                "spacing_z": float(row["spacing_z"]) if row.get("spacing_z") else None,
                "lung_area_px": int(row["lung_area_px"]) if row.get("lung_area_px") else 0,
                "lesion_present": lesion_present,
                "lesion_area_px": int(row["lesion_area_px"]) if row.get("lesion_area_px") else 0,
                "source_volume_path": row.get("source_volume_path", ""),
                "source_mask_path": row.get("source_mask_path", ""),
                "qa_status": row.get("qa_status", "PASSED"),
            }
            test_slices.append(slice_record)

    # 4. Kiểm tra đối chiếu số lượng
    assert len(test_slices) == len(test_slice_ids), (
        f"Lỗi: Số lát cắt trích xuất ({len(test_slices)}) không khớp với split ({len(test_slice_ids)})!"
    )
    assert test_patients_found == test_patients, (
        f"Lỗi: Các bệnh nhân test tìm thấy {test_patients_found} khác dự kiến {test_patients}!"
    )

    stats: Dict[str, Any] = {
        "total_test_patients": len(test_patients),
        "total_test_slices": len(test_slices),
        "lesion_slices": lesion_slices_count,
        "normal_slices": normal_slices_count,
        "patient_slice_counts": patient_slice_counts,
        "patient_lesion_counts": patient_lesion_counts,
    }

    return sorted(list(test_patients)), test_slices, stats


def verify_against_freeze_manifest(
    freeze_manifest_path: Path,
    test_patients: List[str],
    total_test_slices: int,
) -> None:
    """Đối chiếu với bộ dữ liệu đã niêm phong tại DATA-12."""
    if not freeze_manifest_path.exists():
        print(f"[!] Bỏ qua đối chiếu freeze manifest: Không tìm thấy {freeze_manifest_path}")
        return

    with open(freeze_manifest_path, "r", encoding="utf-8") as f:
        freeze_data = json.load(f)

    frozen_test_patients = freeze_data.get("patient_distribution", {}).get("test_patients", [])
    frozen_test_slices = freeze_data.get("slice_distribution", {}).get("test_slices")

    if frozen_test_patients:
        assert sorted(frozen_test_patients) == sorted(test_patients), (
            f"Bất nhất với freeze manifest: frozen {frozen_test_patients} vs hiện tại {test_patients}"
        )
    if frozen_test_slices is not None:
        assert frozen_test_slices == total_test_slices, (
            f"Bất nhất số lát cắt test: frozen {frozen_test_slices} vs hiện tại {total_test_slices}"
        )
    print(f"[+] Đối chiếu thành công với freeze manifest: {freeze_manifest_path.name}")


def make_file_readonly(file_path: Path) -> None:
    """Thiết lập quyền read-only (chỉ đọc) cho file để ngăn chặn vô tình sửa đổi."""
    # 0o444: r--r--r-- (chỉ cho phép đọc cho owner, group và others)
    os.chmod(file_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def make_file_writable(file_path: Path) -> None:
    """Tạm thời mở quyền ghi cho file (để ghi đè khi có cờ --force)."""
    if file_path.exists():
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def is_file_readonly(file_path: Path) -> bool:
    """Kiểm tra file có đang ở trạng thái read-only hay không."""
    return not os.access(file_path, os.W_OK)


def prepare_test_manifest(
    splits_json_path: Path,
    clean_manifest_csv_path: Path,
    output_json_path: Path,
    freeze_manifest_path: Path,
    force: bool = False,
    lock_permissions: bool = True,
) -> Path:
    """Quy trình chính tạo và niêm phong file test_manifest.json."""
    print("=" * 70)
    print("TASK EVAL-01: CHUẨN BỊ VÀ CÔ LẬP TẬP KIỂM THỬ ĐỘC LẬP (INDEPENDENT TEST SET)")
    print("=" * 70)

    # 1. Kiểm tra sự tồn tại của dữ liệu đầu vào
    for p, label in [
        (splits_json_path, "Splits JSON"),
        (clean_manifest_csv_path, "Clean Slices Manifest CSV"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"[X] Không tìm thấy {label} tại: {p}")

    # 2. Xử lý quyền ghi nếu file output đã tồn tại
    if output_json_path.exists():
        if is_file_readonly(output_json_path):
            if force:
                print(f"[*] File {output_json_path} đang bị khóa read-only. Mở khóa do có cờ --force...")
                make_file_writable(output_json_path)
            else:
                raise PermissionError(
                    f"[X] File '{output_json_path}' đang ở chế độ READ-ONLY (đã được niêm phong).\n"
                    f"-> Để ghi đè có chủ đích, hãy chạy lại lệnh với cờ: --force"
                )

    # 3. Đọc dữ liệu
    print(f"-> Đang đọc phân vùng bệnh nhân từ: {splits_json_path}")
    with open(splits_json_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    print(f"-> Đang đọc danh mục lát cắt làm sạch từ: {clean_manifest_csv_path}")
    clean_manifest_rows: List[Dict[str, str]] = []
    with open(clean_manifest_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_manifest_rows.append(row)

    # 4. Xác thực tính cô lập và trích xuất lát cắt tập test
    print("-> Đang kiểm tra tính cô lập dữ liệu (Zero Leakage Verification)...")
    test_patients, test_slices, stats = verify_patient_and_slice_isolation(
        splits_data, clean_manifest_rows
    )

    # 5. Đối chiếu với freeze manifest nếu có
    verify_against_freeze_manifest(freeze_manifest_path, test_patients, len(test_slices))

    # 6. Thu thập mã băm SHA-256 các file nguồn
    sha_clean_manifest = compute_sha256(clean_manifest_csv_path)
    sha_splits_json = compute_sha256(splits_json_path)

    # 7. Xây dựng cấu trúc manifest kiểm thử hoàn chỉnh
    test_manifest_payload: Dict[str, Any] = {
        "metadata": {
            "task_id": "EVAL-01",
            "task_name": "Independent Test Set Isolation & Cataloging",
            "project": "CT Lung Lesion Segmentation via Attention U-Net",
            "dataset_version": "v1.0-frozen",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split_ratio": splits_data.get("metadata", {}).get("split_ratio", "70/15/15"),
            "random_seed": splits_data.get("metadata", {}).get("random_seed", 2026),
            "test_patients": test_patients,
            "total_test_patients": stats["total_test_patients"],
            "total_test_slices": stats["total_test_slices"],
            "lesion_slices": stats["lesion_slices"],
            "normal_slices": stats["normal_slices"],
            "patient_slice_counts": stats["patient_slice_counts"],
            "patient_lesion_counts": stats["patient_lesion_counts"],
            "specifications": {
                "target_resolution": [256, 256],
                "hu_window_range": [-1000.0, 400.0],
                "normalization": "Min-Max [0.0, 1.0]",
                "mask_type": "Binary {0, 1} (1: Lesion/Infection, lesion_label=3)",
                "interpolation_image": "Bilinear (cv2.INTER_LINEAR)",
                "interpolation_mask": "Nearest Neighbor (cv2.INTER_NEAREST)",
                "batch_size": 1,
                "shuffle": False,
                "augmentation": False,
            },
            "source_files": {
                "splits_json": str(splits_json_path),
                "clean_slices_csv": str(clean_manifest_csv_path),
                "freeze_manifest": str(freeze_manifest_path) if freeze_manifest_path.exists() else None,
            },
            "source_checksums": {
                "clean_slices_manifest_csv_sha256": sha_clean_manifest,
                "splits_patient_level_json_sha256": sha_splits_json,
            },
            "qa_status": {
                "zero_leakage_verified": True,
                "patient_level_isolation_verified": True,
                "read_only_enforced": lock_permissions,
            },
        },
        "slices": test_slices,
    }

    # 8. Ghi dữ liệu ra file output_json_path
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(test_manifest_payload, f, indent=2, ensure_ascii=False)
    print(f"[+] Đã ghi thành công manifest tập test ra: {output_json_path}")

    # 9. Khóa quyền ghi (chỉ đọc - read only)
    if lock_permissions:
        make_file_readonly(output_json_path)
        is_locked = is_file_readonly(output_json_path)
        print(f"[+] Đã thiết lập trạng thái READ-ONLY (0o444) cho file: {output_json_path} (Locked: {is_locked})")
    else:
        print(f"[!] Cảnh báo: File không được khóa quyền ghi (--no-lock được kích hoạt).")

    # 10. In bảng tóm tắt nghiệm thu
    print("-" * 70)
    print("TỔNG HỢP KẾT QUẢ CÔ LẬP TẬP TEST ĐỘC LẬP:")
    print(f"  * Danh sách bệnh nhân Test ({stats['total_test_patients']} ca): {', '.join(test_patients)}")
    print(f"  * Tổng số lát cắt kiểm thử : {stats['total_test_slices']} lát cắt")
    print(f"    - Lát cắt có tổn thương : {stats['lesion_slices']} ({stats['lesion_slices']/stats['total_test_slices']*100:.1f}%)")
    print(f"    - Lát cắt bình thường   : {stats['normal_slices']} ({stats['normal_slices']/stats['total_test_slices']*100:.1f}%)")
    print("  * Phân bổ lát cắt theo từng ca bệnh:")
    for pid, c in stats["patient_slice_counts"].items():
        les = stats["patient_lesion_counts"][pid]["lesion_slices"]
        nor = stats["patient_lesion_counts"][pid]["normal_slices"]
        print(f"    - {pid:26s}: {c:3d} lát cắt (Tổn thương: {les:3d}, Bình thường: {nor:3d})")
    print("=" * 70)

    return output_json_path


def main():
    parser = argparse.ArgumentParser(
        description="EVAL-01: Chuẩn bị và cô lập tập kiểm thử độc lập (Independent Test Set)."
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path("data/processed/splits_patient_level.json"),
        help="Đường dẫn file splits_patient_level.json",
    )
    parser.add_argument(
        "--clean-manifest-csv",
        type=Path,
        default=Path("data/processed/clean_slices_manifest.csv"),
        help="Đường dẫn file clean_slices_manifest.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/test_manifest.json"),
        help="Đường dẫn lưu file test_manifest.json đầu ra",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("data/processed/dataset_v1.0_freeze_manifest.json"),
        help="Đường dẫn file dataset_v1.0_freeze_manifest.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Cho phép mở khóa file và ghi đè nếu file output đã tồn tại ở chế độ read-only",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Không khóa quyền ghi (không gán 0o444) sau khi tạo file",
    )

    args = parser.parse_args()

    prepare_test_manifest(
        splits_json_path=args.splits_json,
        clean_manifest_csv_path=args.clean_manifest_csv,
        output_json_path=args.output_json,
        freeze_manifest_path=args.freeze_manifest,
        force=args.force,
        lock_permissions=not args.no_lock,
    )


if __name__ == "__main__":
    main()
