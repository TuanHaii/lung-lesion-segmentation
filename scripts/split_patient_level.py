import csv
import json
from pathlib import Path
import numpy as np


def main():
    manifest_file = Path("data/processed/clean_slices_manifest.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_output = output_dir / "splits_patient_level.json"
    csv_summary_output = output_dir / "summary_split.csv"

    if not manifest_file.exists():
        print(f"Lỗi: Không tìm thấy file {manifest_file} từ task DATA-04!")
        return

    # 1. Đọc manifest lát cắt
    records = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Đã đọc {len(records)} lát cắt chuẩn hóa từ manifest.")

    # 2. Gom nhóm lát cắt theo từng bệnh nhân
    patient_slices = {}
    for r in records:
        pid = r["patient_id"]
        if pid not in patient_slices:
            patient_slices[pid] = []
        patient_slices[pid].append(r)

    unique_patients = sorted(list(patient_slices.keys()))
    total_patients = len(unique_patients)
    print(f"Tổng số bệnh nhân phát hiện: {total_patients}")

    if total_patients != 20:
        print(f"Cảnh báo: Số ca bệnh hiện tại là {total_patients} (kế hoạch chuẩn là 20 ca).")

    # 3. Phân chia Patient-level với seed cố định = 2026
    # Phân bổ tỷ lệ 70% / 15% / 15% -> 14 Train / 3 Val / 3 Test
    np.random.seed(2026)
    shuffled_patients = np.random.permutation(unique_patients).tolist()

    n_train = int(round(0.70 * total_patients))  # 14 ca
    n_val = int(round(0.15 * total_patients))    # 3 ca
    # Số ca còn lại thuộc test set (3 ca)
    
    train_patients = sorted(shuffled_patients[:n_train])
    val_patients = sorted(shuffled_patients[n_train : n_train + n_val])
    test_patients = sorted(shuffled_patients[n_train + n_val :])

    # 4. Kiểm tra giao thoa giữa các tập (Zero-leakage verification)
    set_train = set(train_patients)
    set_val = set(val_patients)
    set_test = set(test_patients)

    assert set_train.isdisjoint(set_val), "Rò rỉ dữ liệu phát hiện giữa Train và Val!"
    assert set_train.isdisjoint(set_test), "Rò rỉ dữ liệu phát hiện giữa Train và Test!"
    assert set_val.isdisjoint(set_test), "Rò rỉ dữ liệu phát hiện giữa Val và Test!"

    splits = {
        "metadata": {
            "random_seed": 2026,
            "split_ratio": "70/15/15",
            "total_patients": total_patients,
            "total_slices": len(records),
        },
        "patient_splits": {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
        },
        "slice_samples": {
            "train": [s["sample_id"] for p in train_patients for s in patient_slices[p]],
            "val": [s["sample_id"] for p in val_patients for s in patient_slices[p]],
            "test": [s["sample_id"] for p in test_patients for s in patient_slices[p]],
        },
    }

    # 5. Xuất file JSON phân bổ
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # 6. Tổng hợp số liệu thống kê ra summary_split.csv
    summary_rows = []
    for split_name, p_list in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
        all_s = [s for p in p_list for s in patient_slices[p]]
        lesion_s = [s for s in all_s if int(s["lesion_present"]) == 1]
        normal_s = [s for s in all_s if int(s["lesion_present"]) == 0]
        
        summary_rows.append({
            "split": split_name,
            "num_patients": len(p_list),
            "patient_ids": ";".join(p_list),
            "total_slices": len(all_s),
            "lesion_slices": len(lesion_s),
            "normal_slices": len(normal_s),
            "slice_percentage": f"{len(all_s) / len(records) * 100:.2f}%",
            "lesion_ratio": f"{len(lesion_s) / len(all_s) * 100:.2f}%" if all_s else "0%",
        })

    with open(csv_summary_output, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "num_patients",
            "total_slices",
            "lesion_slices",
            "normal_slices",
            "slice_percentage",
            "lesion_ratio",
            "patient_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # In kết quả kiểm toán
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ PHÂN CHIA DỮ LIỆU PATIENT-LEVEL (SEED = 2026)")
    print("=" * 70)
    for row in summary_rows:
        print(
            f"[{row['split'].upper():<5}] {row['num_patients']} bệnh nhân | "
            f"Tổng: {row['total_slices']} slices ({row['slice_percentage']}) | "
            f"Tổn thương: {row['lesion_slices']} | Lành: {row['normal_slices']}"
        )
    print("=" * 70)
    print(f"=> Đã lưu JSON phân bổ: {json_output}")
    print(f"=> Đã lưu bảng thống kê: {csv_summary_output}")


if __name__ == "__main__":
    main()