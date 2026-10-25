"""
Task ID: EVAL-02 / EVAL-06 (Sprint W5 - Model Comparison)
Mô tả: Script tự động tổng hợp bảng số liệu báo cáo so sánh các mô hình
       (Attention U-Net vs Standard U-Net vs SegNet) từ các file CSV kết quả
       trong thư mục reports/.
Tuân thủ Dual-reporting strategy:
- Chế độ 1: Lesion-containing Slices (GT > 0) [Báo cáo chính trong bài báo]
- Chế độ 2: All Slices [Báo cáo bổ trợ]
- Chế độ 3: Patient-level Volume Aggregation
Xuất ra màn hình console (Markdown), file CSV tổng hợp và mã bảng LaTeX.
"""

import os
import sys
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


def mean_and_std(values: List[float]) -> tuple[float, float]:
    """Tính trung bình (mean) và độ lệch chuẩn mẫu (sample standard deviation)."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    if n <= 1:
        return m, 0.0
    variance = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(variance)


def load_slice_metrics(slice_csv_path: str) -> Dict[str, Any]:
    """Đọc và trích xuất chỉ số từ test_metrics_slice_level.csv."""
    records = []
    with open(slice_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "sample_id": row.get("sample_id", ""),
                "patient_id": row.get("patient_id", ""),
                "slice_index": int(row.get("slice_index", 0)),
                "lesion_present": int(float(row.get("lesion_present", 0))),
                "dice": float(row.get("dice", 0.0)),
                "iou": float(row.get("iou", 0.0)),
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
            })

    # Chế độ 1: Lesion-containing Slices (GT > 0)
    lesion_slices = [r for r in records if r["lesion_present"] == 1]
    # Chế độ 2: All Slices
    all_slices = records

    return {
        "total_slices": len(all_slices),
        "lesion_slices_count": len(lesion_slices),
        "lesion_dice_list": [r["dice"] for r in lesion_slices],
        "lesion_iou_list": [r["iou"] for r in lesion_slices],
        "lesion_precision_list": [r["precision"] for r in lesion_slices],
        "lesion_recall_list": [r["recall"] for r in lesion_slices],
        "all_dice_list": [r["dice"] for r in all_slices],
        "all_iou_list": [r["iou"] for r in all_slices],
        "all_precision_list": [r["precision"] for r in all_slices],
        "all_recall_list": [r["recall"] for r in all_slices],
    }


def load_patient_metrics(patient_csv_path: str) -> Dict[str, Any]:
    """Đọc và trích xuất chỉ số từ test_metrics_patient_level.csv."""
    records = []
    with open(patient_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "patient_id": row.get("patient_id", ""),
                "patient_dice": float(row.get("patient_dice", 0.0)),
                "patient_iou": float(row.get("patient_iou", 0.0)),
                "patient_precision": float(row.get("patient_precision", 0.0)),
                "patient_recall": float(row.get("patient_recall", 0.0)),
            })

    return {
        "patient_count": len(records),
        "patient_dice_list": [r["patient_dice"] for r in records],
        "patient_iou_list": [r["patient_iou"] for r in records],
        "patient_precision_list": [r["patient_precision"] for r in records],
        "patient_recall_list": [r["patient_recall"] for r in records],
        "patient_records": {r["patient_id"]: r for r in records},
    }


def find_model_reports(base_reports_dir: str) -> Dict[str, Dict[str, str]]:
    """Tự động quét và tìm các cặp file CSV của từng mô hình."""
    base_dir = Path(base_reports_dir)
    models = {}

    if not base_dir.exists():
        return models

    # Kiểm tra các thư mục con trong reports/ (ví dụ reports/attunet, reports/unet, reports/segnet)
    for p in base_dir.iterdir():
        if p.is_dir():
            slice_csv = p / "test_metrics_slice_level.csv"
            pat_csv = p / "test_metrics_patient_level.csv"
            if slice_csv.exists() and pat_csv.exists():
                models[p.name] = {
                    "name": p.name,
                    "slice_csv": str(slice_csv),
                    "patient_csv": str(pat_csv)
                }

    # Nếu file nằm ngay tại thư mục reports gốc
    root_slice = base_dir / "test_metrics_slice_level.csv"
    root_pat = base_dir / "test_metrics_patient_level.csv"
    if root_slice.exists() and root_pat.exists() and "default" not in models:
        models["Default_Model"] = {
            "name": "Default_Model",
            "slice_csv": str(root_slice),
            "patient_csv": str(root_pat)
        }

    return models


def generate_comparison_report(reports_dir: str = "reports", output_csv: Optional[str] = None):
    """Tổng hợp và in báo cáo so sánh mô hình."""
    model_paths = find_model_reports(reports_dir)

    if not model_paths:
        print(f"[!] Không tìm thấy kết quả đánh giá nào trong thư mục '{reports_dir}'!")
        print("    Cấu trúc mong đợi: reports/<model_name>/test_metrics_slice_level.csv")
        print("    Ví dụ: reports/attunet/, reports/unet/, reports/segnet/")
        return

    # Chuẩn hóa tên hiển thị
    display_name_map = {
        "attunet": "Attention U-Net (Proposed)",
        "attention_unet": "Attention U-Net (Proposed)",
        "attentionunet": "Attention U-Net (Proposed)",
        "exp-attunet-001": "Attention U-Net (Proposed)",
        "unet": "Standard U-Net (Baseline)",
        "baseline_unet": "Standard U-Net (Baseline)",
        "exp-unet-001": "Standard U-Net (Baseline)",
        "segnet": "SegNet (Baseline)",
        "exp-segnet-001": "SegNet (Baseline)",
    }

    results = []
    raw_data_store = {}

    for folder_name, paths in model_paths.items():
        norm_key = folder_name.lower()
        model_title = display_name_map.get(norm_key, folder_name)

        slice_data = load_slice_metrics(paths["slice_csv"])
        pat_data = load_patient_metrics(paths["patient_csv"])

        # Chế độ 1: Lesion slices
        l_dice_m, l_dice_s = mean_and_std(slice_data["lesion_dice_list"])
        l_iou_m, l_iou_s = mean_and_std(slice_data["lesion_iou_list"])
        l_prec_m, l_prec_s = mean_and_std(slice_data["lesion_precision_list"])
        l_rec_m, l_rec_s = mean_and_std(slice_data["lesion_recall_list"])

        # Chế độ 2: All slices
        a_dice_m, a_dice_s = mean_and_std(slice_data["all_dice_list"])
        a_iou_m, a_iou_s = mean_and_std(slice_data["all_iou_list"])

        # Chế độ 3: Patient level
        p_dice_m, p_dice_s = mean_and_std(pat_data["patient_dice_list"])
        p_iou_m, p_iou_s = mean_and_std(pat_data["patient_iou_list"])

        row = {
            "model_key": folder_name,
            "model_name": model_title,
            "n_lesion_slices": slice_data["lesion_slices_count"],
            "lesion_dice": f"{l_dice_m:.4f} ± {l_dice_s:.4f}",
            "lesion_iou": f"{l_iou_m:.4f} ± {l_iou_s:.4f}",
            "lesion_prec": f"{l_prec_m:.4f} ± {l_prec_s:.4f}",
            "lesion_rec": f"{l_rec_m:.4f} ± {l_rec_s:.4f}",
            "all_dice": f"{a_dice_m:.4f} ± {a_dice_s:.4f}",
            "all_iou": f"{a_iou_m:.4f} ± {a_iou_s:.4f}",
            "patient_dice": f"{p_dice_m:.4f} ± {p_dice_s:.4f}",
            "patient_iou": f"{p_iou_m:.4f} ± {p_iou_s:.4f}",
            "raw_l_dice_m": l_dice_m,
            "raw_l_iou_m": l_iou_m,
            "raw_p_dice_m": p_dice_m,
        }
        results.append(row)
        raw_data_store[folder_name] = {
            "lesion_dice_list": slice_data["lesion_dice_list"],
            "patient_dice_list": pat_data["patient_dice_list"],
        }

    # Sắp xếp ưu tiên: Proposed (Attention U-Net) lên đầu, tiếp đến UNet, SegNet
    def sort_order(x):
        title = x["model_name"].lower()
        if "proposed" in title or "att" in title:
            return 0
        if "unet" in title:
            return 1
        if "segnet" in title:
            return 2
        return 3

    results.sort(key=sort_order)

    # 1. In Bảng Markdown ra Console
    print("\n" + "=" * 90)
    print(" BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH PHÂN ĐOẠN TỔN THƯƠNG CT PHỔI (SPRINT W5)")
    print("=" * 90)
    print("\n### BẢNG 1: BÁO CÁO CHÍNH - CÁC LÁT CẮT CÓ TỔN THƯƠNG (LESION-CONTAINING SLICES, GT > 0)\n")
    print(f"| {'Mô hình (Model)':<30} | {'Dice (DSC)':<17} | {'IoU (Jaccard)':<17} | {'Precision':<17} | {'Recall':<17} |")
    print(f"|:{'-'*30}-|-{'-'*17}:|-{'-'*17}:|-{'-'*17}:|-{'-'*17}:|")
    for r in results:
        print(f"| {r['model_name']:<30} | {r['lesion_dice']:<17} | {r['lesion_iou']:<17} | {r['lesion_prec']:<17} | {r['lesion_rec']:<17} |")

    print("\n### BẢNG 2: BÁO CÁO TỔNG HỢP CẤP BỆNH NHÂN & TOÀN BỘ SLICES (PATIENT-LEVEL & ALL SLICES)\n")
    print(f"| {'Mô hình (Model)':<30} | {'Patient DSC':<17} | {'Patient IoU':<17} | {'All-Slices DSC':<17} | {'All-Slices IoU':<17} |")
    print(f"|:{'-'*30}-|-{'-'*17}:|-{'-'*17}:|-{'-'*17}:|-{'-'*17}:|")
    for r in results:
        print(f"| {r['model_name']:<30} | {r['patient_dice']:<17} | {r['patient_iou']:<17} | {r['all_dice']:<17} | {r['all_iou']:<17} |")

    # 2. Kiểm định thống kê Wilcoxon Signed-Rank Test nếu có ít nhất 2 mô hình
    if len(results) >= 2 and wilcoxon is not None:
        proposed_key = results[0]["model_key"]
        print("\n### KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (WILCOXON SIGNED-RANK TEST, p < 0.05)")
        for other in results[1:]:
            other_key = other["model_key"]
            l1 = raw_data_store[proposed_key]["lesion_dice_list"]
            l2 = raw_data_store[other_key]["lesion_dice_list"]
            if len(l1) == len(l2) and len(l1) > 0:
                try:
                    stat, p_val = wilcoxon(l1, l2)
                    sig_text = "Có ý nghĩa thống kê (p < 0.05)" if p_val < 0.05 else "Chưa có ý nghĩa thống kê (p >= 0.05)"
                    print(f" - {results[0]['model_name']} vs {other['model_name']}: W={stat:.1f}, p-value={p_val:.4e} -> {sig_text}")
                except Exception as e:
                    pass

    # 3. Xuất file CSV
    if output_csv is None:
        output_csv = os.path.join(reports_dir, "model_comparison_summary.csv")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_name", "n_lesion_slices", "lesion_dice", "lesion_iou",
            "lesion_prec", "lesion_rec", "patient_dice", "patient_iou",
            "all_slices_dice", "all_slices_iou"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model_name": r["model_name"],
                "n_lesion_slices": r["n_lesion_slices"],
                "lesion_dice": r["lesion_dice"],
                "lesion_iou": r["lesion_iou"],
                "lesion_prec": r["lesion_prec"],
                "lesion_rec": r["lesion_rec"],
                "patient_dice": r["patient_dice"],
                "patient_iou": r["patient_iou"],
                "all_slices_dice": r["all_dice"],
                "all_slices_iou": r["all_iou"],
            })
    print(f"\n[+] Đã lưu bảng tổng hợp CSV tại: {output_csv}")

    # 4. Xuất mã LaTeX cho Bài báo / Báo cáo
    latex_path = os.path.join(reports_dir, "model_comparison_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Bảng so sánh mô hình cho bài báo IEEE\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Quantitative Segmentation Performance Comparison on Independent Test Set}\n")
        f.write("\\label{tab:model_comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Dice (DSC)} & \\textbf{IoU (Jaccard)} & \\textbf{Precision} & \\textbf{Recall} \\\\\n")
        f.write("\\hline\n")
        for r in results:
            m_label = "\\textbf{" + r['model_name'] + "}" if "Proposed" in r['model_name'] else r['model_name']
            f.write(f"{m_label} & {r['lesion_dice']} & {r['lesion_iou']} & {r['lesion_prec']} & {r['lesion_rec']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[+] Đã lưu mã bảng LaTeX tại : {latex_path}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and Compare Segmentation Models from Evaluation Reports")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory containing model evaluation reports")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save summary CSV")
    args = parser.parse_args()

    generate_comparison_report(reports_dir=args.reports_dir, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
