import json
from pathlib import Path
import nibabel as nib
import numpy as np

def clean_stem(filename: str) -> str:
    # Lược bỏ các phần mở rộng và hậu tố mask để ghép cặp ID chính xác
    name = filename.replace(".nii.gz", "").replace(".nii", "")
    return name.replace("_mask", "").replace("_infection", "")

def main():
    ct_dir = Path("data/raw/ct_volumes")
    mask_dir = Path("data/raw/ground_truth_masks")
    output_dir = Path("reports/DATA-03")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "Raw_Data_Integrity_Report.json"

    ct_files = sorted(list(ct_dir.glob("*.nii*")))
    mask_files = sorted(list(mask_dir.glob("*.nii*")))

    if not ct_files:
        print("Lỗi: Không tìm thấy file trong data/raw/ct_volumes/")
        return

    report = {
        "task_id": "DATA-03",
        "total_ct_volumes": len(ct_files),
        "total_mask_volumes": len(mask_files),
        "volumes": {}
    }

    print(f"Bắt đầu quét metadata cho {len(ct_files)} volumes...")

    for ct_path in ct_files:
        case_id = clean_stem(ct_path.name)
        
        # Đọc dữ liệu CT
        ct_nii = nib.load(str(ct_path))
        ct_header = ct_nii.header
        ct_data = ct_nii.get_fdata(dtype=np.float32)
        
        # Tìm file mask tương ứng
        matched_masks = [m for m in mask_files if clean_stem(m.name) == case_id]
        mask_info = {"matched": False}
        
        if matched_masks:
            mask_path = matched_masks[0]
            mask_nii = nib.load(str(mask_path))
            mask_data = mask_nii.get_fdata(dtype=np.float32)
            mask_unique_values = [int(v) for v in np.unique(mask_data)]
            
            mask_info = {
                "matched": True,
                "mask_filename": mask_path.name,
                "mask_shape": list(mask_data.shape),
                "unique_labels": mask_unique_values,
                "shape_matched": list(ct_data.shape) == list(mask_data.shape)
            }
        
        report["volumes"][case_id] = {
            "ct_filename": ct_path.name,
            "image_shape": list(ct_data.shape),
            "num_slices": int(ct_data.shape[2]) if len(ct_data.shape) == 3 else int(ct_data.shape[-1]),
            "voxel_spacing_mm": [round(float(x), 4) for x in ct_header.get_zooms()[:3]],
            "raw_intensity_range": [round(float(np.min(ct_data)), 2), round(float(np.max(ct_data)), 2)],
            "mask_verification": mask_info
        }
        
        status = "KHỚP" if mask_info.get("shape_matched") else "LỆCH/THIẾU"
        print(f"[{status}] {case_id} | Shape: {ct_data.shape} | Spacing: {ct_header.get_zooms()[:3]} | HU: [{np.min(ct_data):.1f}, {np.max(ct_data):.1f}]")

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n=> Hoàn tất DATA-03! Báo cáo metadata đã xuất tại: {report_file}")

if __name__ == "__main__":
    main()
