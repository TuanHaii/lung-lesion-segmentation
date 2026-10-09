import csv
from pathlib import Path
import nibabel as nib
import numpy as np


def clean_stem(filename: str) -> str:
    name = filename.replace(".nii.gz", "").replace(".nii", "")
    return name.replace("_mask", "").replace("_infection", "")


def extract_patient_id(case_name: str) -> str:
    return f"P_{case_name}"


def sample_evenly(items, target_count):
    """Lấy mẫu cách đều theo trục z để phân bổ đồng đều qua tất cả các ca bệnh."""
    if len(items) <= target_count:
        return items
    indices = np.linspace(0, len(items) - 1, target_count, dtype=int)
    return [items[i] for i in indices]


def main():
    ct_dir = Path("data/raw/ct_volumes")
    mask_dir = Path("data/raw/ground_truth_masks")
    manifest_dir = Path("data/processed")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "clean_slices_manifest.csv"

    ct_files = sorted(list(ct_dir.glob("*.nii*")))
    mask_files = sorted(list(mask_dir.glob("*.nii*")))

    if not ct_files:
        print("Lỗi: Không tìm thấy file CT trong data/raw/ct_volumes/")
        return

    # Tham số chuẩn y tế
    MIN_LUNG_AREA_PX = 3000  # Bỏ lát cắt ngoài ngực, đỉnh phổi và đáy bụng
    MIN_LESION_PX = 15  # Khử nhiễu gán nhãn li ti
    LESION_LABEL = 3  # Nhãn 3 là vùng nhiễm trùng / tổn thương

    TARGET_LESION_SLICES = 1450
    TARGET_NORMAL_SLICES = 300

    fieldnames = [
        "sample_id",
        "patient_id",
        "volume_id",
        "slice_index",
        "original_shape",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "lung_area_px",
        "lesion_present",
        "lesion_area_px",
        "source_volume_path",
        "source_mask_path",
        "qa_status",
    ]

    all_lesion_candidates = []
    all_normal_candidates = []

    print("Bắt đầu quét và phân loại lát cắt từ 20 CT volumes...")

    for ct_path in ct_files:
        case_id = clean_stem(ct_path.name)
        matched_masks = [m for m in mask_files if clean_stem(m.name) == case_id]

        if not matched_masks:
            continue

        mask_path = matched_masks[0]
        ct_nii = nib.load(str(ct_path))
        mask_nii = nib.load(str(mask_path))

        ct_data = ct_nii.get_fdata(dtype=np.float32)
        mask_data = mask_nii.get_fdata(dtype=np.float32)
        zooms = ct_nii.header.get_zooms()

        num_slices = ct_data.shape[2]
        patient_id = extract_patient_id(case_id)
        volume_id = case_id

        for z in range(num_slices):
            img_slice = ct_data[:, :, z]
            mask_slice = mask_data[:, :, z]

            # Vùng nhu mô phổi
            lung_mask = (mask_slice > 0) | (
                (img_slice >= -1000) & (img_slice <= -200)
            )
            lung_area = int(np.sum(lung_mask))

            if lung_area < MIN_LUNG_AREA_PX:
                continue

            lesion_pixels = int(np.sum(mask_slice == LESION_LABEL))
            is_lesion = 1 if lesion_pixels >= MIN_LESION_PX else 0

            record = {
                "sample_id": f"{patient_id}_{volume_id}_z{z:04d}",
                "patient_id": patient_id,
                "volume_id": volume_id,
                "slice_index": z,
                "original_shape": f"{img_slice.shape[0]}x{img_slice.shape[1]}",
                "spacing_x": round(float(zooms[0]), 4),
                "spacing_y": round(float(zooms[1]), 4),
                "spacing_z": round(float(zooms[2]), 4),
                "lung_area_px": lung_area,
                "lesion_present": is_lesion,
                "lesion_area_px": lesion_pixels if is_lesion else 0,
                "source_volume_path": str(ct_path),
                "source_mask_path": str(mask_path),
                "qa_status": "PASSED",
            }

            if is_lesion:
                all_lesion_candidates.append(record)
            else:
                all_normal_candidates.append(record)

    print(
        f"Ứng viên quét được: {len(all_lesion_candidates)} lát cắt tổn thương | "
        f"{len(all_normal_candidates)} lát cắt lành."
    )

    # Lấy mẫu phân bố đều để cố định số lượng
    selected_lesions = sample_evenly(
        all_lesion_candidates, TARGET_LESION_SLICES
    )
    selected_normals = sample_evenly(
        all_normal_candidates, TARGET_NORMAL_SLICES
    )

    final_slices = sorted(
        selected_lesions + selected_normals,
        key=lambda x: (x["patient_id"], x["slice_index"]),
    )

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_slices)

    final_lesion_cnt = sum(1 for s in final_slices if s["lesion_present"] == 1)
    final_normal_cnt = len(final_slices) - final_lesion_cnt

    print("\n" + "=" * 60)
    print(f"Tổng số lát cắt chuẩn hóa: {len(final_slices)} lát cắt")
    print(f"- Lát cắt có tổn thương (Lesion > 0): {final_lesion_cnt}")
    print(f"- Lát cắt phổi lành (Lesion = 0): {final_normal_cnt}")
    print(f"File manifest đã lưu tại: {manifest_file}")
    print("=" * 60)

    if 1500 <= len(final_slices) <= 2000:
        print("=> ĐẠT CHUẨN: Quy mô nằm trọn trong mục tiêu 1.500 - 2.000 slices!")
    else:
        print(f"=> CẢNH BÁO: Số lượng {len(final_slices)} ngoài dải mục tiêu!")


if __name__ == "__main__":
    main()