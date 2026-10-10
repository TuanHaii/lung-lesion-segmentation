import os
import json
import random
from collections import defaultdict

def create_patient_level_splits(
    data_dir="data/processed",
    output_json="data/splits/splits_patient_level.json",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Quét danh sách file trong data_dir, gom nhóm theo Patient ID
    và chia tập Train / Val / Test theo cấp độ Bệnh nhân.
    """
    random.seed(seed)
    
    if not os.path.exists(data_dir):
        print(f"Thư mục '{data_dir}' chưa tồn tại. Đang tạo thư mục mô phỏng...")
        os.makedirs(data_dir, exist_ok=True)
        # Tạo dữ liệu giả lập để test script nếu chưa có data thực
        for p_id in range(1, 21): # 20 bệnh nhân
            for slice_idx in range(1, 10): # Mỗi bệnh nhân 9 slices
                filename = f"patient_{p_id:03d}_slice_{slice_idx:02d}.npy"
                open(os.path.join(data_dir, filename), 'a').close()

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # 1. Gom nhóm file theo Patient ID
    # Giả định định dạng file: patient_XXX_slice_YY.npy hoặc XXX_slice_YY.npy
    patient_to_files = defaultdict(list)
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy') or f.endswith('.png')]

    for filename in all_files:
        # Lấy ID bệnh nhân (ví dụ: 'patient_001_slice_01.npy' -> 'patient_001')
        patient_id = filename.split('_slice_')[0] if '_slice_' in filename else filename.split('_')[0]
        patient_to_files[patient_id].append(filename)

    unique_patients = list(patient_to_files.keys())
    random.shuffle(unique_patients)

    num_patients = len(unique_patients)
    num_train = int(num_patients * train_ratio)
    num_val = int(num_patients * val_ratio)

    train_patients = unique_patients[:num_train]
    val_patients = unique_patients[num_train:num_train + num_val]
    test_patients = unique_patients[num_train + num_val:]

    # 2. Tổng hợp danh sách file cho từng tập
    splits = {
        "train": [f for p in train_patients for f in patient_to_files[p]],
        "val": [f for p in val_patients for f in patient_to_files[p]],
        "test": [f for p in test_patients for f in patient_to_files[p]],
        "patient_distribution": {
            "train_patients": train_patients,
            "val_patients": val_patients,
            "test_patients": test_patients
        }
    }

    # 3. Lưu thông tin split ra file JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    print("=" * 60)
    print("PATIENT-LEVEL DATA SPLIT COMPLETED")
    print("=" * 60)
    print(f"Tổng số Bệnh nhân  : {num_patients}")
    print(f"Tập Train          : {len(train_patients)} Bệnh nhân ({len(splits['train'])} slices)")
    print(f"Tập Validation     : {len(val_patients)} Bệnh nhân ({len(splits['val'])} slices)")
    print(f"Tập Test           : {len(test_patients)} Bệnh nhân ({len(splits['test'])} slices)")
    print(f"File lưu cấu hình  : {output_json}")
    
    # 4. Kiểm tra Data Leakage
    train_set_p = set(train_patients)
    val_set_p = set(val_patients)
    test_set_p = set(test_patients)
    
    assert len(train_set_p.intersection(val_set_p)) == 0, "LỖI: Rò rỉ giữa Train và Val!"
    assert len(train_set_p.intersection(test_set_p)) == 0, "LỖI: Rò rỉ giữa Train và Test!"
    assert len(val_set_p.intersection(test_set_p)) == 0, "LỖI: Rò rỉ giữa Val và Test!"
    print(">> DATA LEAKAGE CHECK PASSED: KHÔNG CÓ BỆNH NHÂN BỊ TRÙNG LẶP GIAO TẬP <<")
    print("=" * 60)

if __name__ == "__main__":
    create_patient_level_splits()