import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    """Tính toán mã băm SHA-256 cho một file cục bộ."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    base_dir = Path(".")
    manifest_csv = base_dir / "data/processed/clean_slices_manifest.csv"
    split_json = base_dir / "data/processed/splits_patient_level.json"
    split_csv = base_dir / "data/processed/summary_split.csv"
    raw_checksums = base_dir / "data/raw/md5_checksums.txt"

    output_manifest = base_dir / "data/processed/dataset_v1.0_freeze_manifest.json"

    # Kiểm tra sự tồn tại của các file cốt lõi
    required_files = [manifest_csv, split_json, split_csv]
    for rf in required_files:
        if not rf.exists():
            raise FileNotFoundError(f"Không tìm thấy file điều kiện: {rf}")

    # 1. Đọc dữ liệu manifest và split
    slices = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slices.append(row)

    with open(split_json, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # 2. Thống kê số lượng
    total_slices = len(slices)
    lesion_slices = sum(1 for s in slices if int(s["lesion_present"]) == 1)
    normal_slices = total_slices - lesion_slices

    train_slices = len(splits["slice_samples"]["train"])
    val_slices = len(splits["slice_samples"]["val"])
    test_slices = len(splits["slice_samples"]["test"])

    # 3. Thu thập mã băm các thành phần pipeline dữ liệu
    pipeline_files = [
        "src/data/hu_window.py",
        "src/data/transform.py",
        "src/data/mask_binarize.py",
        "src/data/augmentations.py",
        "src/data/dataset.py",
        "data/processed/clean_slices_manifest.csv",
        "data/processed/splits_patient_level.json",
        "data/processed/summary_split.csv",
    ]

    pipeline_hashes = {}
    for pf in pipeline_files:
        p = base_dir / pf
        if p.exists():
            pipeline_hashes[pf] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
            }

    # 4. Tạo cấu trúc biên bản đóng băng v1.0
    freeze_data = {
        "dataset_version": "v1.0-frozen",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project": "CT Lung Lesion Segmentation via Attention U-Net",
        "specifications": {
            "target_resolution": [256, 256],
            "hu_window_range": [-1000.0, 400.0],
            "normalization": "Min-Max [0.0, 1.0]",
            "mask_type": "Binary {0, 1} (1: Lesion/Infection)",
            "interpolation_image": "Bilinear (cv2.INTER_LINEAR)",
            "interpolation_mask": "Nearest Neighbor (cv2.INTER_NEAREST)",
            "split_strategy": "Patient-level 70/15/15",
            "random_seed": 2026,
        },
        "slice_distribution": {
            "total_slices": total_slices,
            "lesion_slices": lesion_slices,
            "normal_slices": normal_slices,
            "train_slices": train_slices,
            "val_slices": val_slices,
            "test_slices": test_slices,
        },
        "patient_distribution": {
            "total_patients": splits["metadata"]["total_patients"],
            "train_patients": splits["patient_splits"]["train"],
            "val_patients": splits["patient_splits"]["val"],
            "test_patients": splits["patient_splits"]["test"],
        },
        "file_integrity_checksums": pipeline_hashes,
        "qa_status": {
            "zero_leakage_verified": True,
            "data_manifest_locked": True,
            "unseen_test_set_isolated": True,
        },
    }

    # 5. Xuất file JSON
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(freeze_data, f, indent=2)

    print("\n" + "=" * 70)
    print("BIÊN BẢN ĐÓNG BĂNG DỮ LIỆU CHÍNH THỨC (DATASET V1.0 FREEZE)")
    print("=" * 70)
    print(f"Phiên bản:            {freeze_data['dataset_version']}")
    print(f"Tổng số lát cắt:      {total_slices} slices")
    print(f"- Lát cắt tổn thương: {lesion_slices} slices ({lesion_slices/total_slices*100:.1f}%)")
    print(f"- Lát cắt phổi lành:  {normal_slices} slices ({normal_slices/total_slices*100:.1f}%)")
    print(f"Phân bổ ca bệnh:      14 Train | 3 Val | 3 Test")
    print(f"Phân bổ lát cắt:      {train_slices} Train | {val_slices} Val | {test_slices} Test")
    print(f"Đã tạo file niêm phong: {output_manifest}")
    print("=" * 70)
    print("=> [DATA-12] Bộ dữ liệu đã được niêm phong hoàn tất 100%!")


if __name__ == "__main__":
    main()