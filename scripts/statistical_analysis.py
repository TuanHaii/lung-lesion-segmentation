#!/usr/bin/env python3
"""
Task ID: EVAL-07 (Đường găng - Data & Evaluation Lead)
Tên script: scripts/statistical_analysis.py
Mục đích:
- Phân tích thống kê định lượng và kiểm định giả thuyết cặp (Statistical Hypothesis Testing)
  giữa Attention U-Net (Proposed) và 2 mô hình cơ sở: Standard U-Net (Baseline), SegNet (Baseline).
- Chiến lược báo cáo song song (Dual-Reporting):
  + Chế độ chính (Primary Mode - in bài báo IEEE): Chỉ tính trên lát cắt chứa tổn thương (GT > 0, N=221).
  + Chế độ bổ trợ (Secondary Mode): Tính trên toàn bộ lát cắt trong test set (All Slices, N=265).
- Tính toán đầy đủ 4 chỉ số y tế: DSC, IoU, Precision, Recall:
  + Mean ± Std.
  + Khoảng tin cậy 95% (95% CI = 1.96 * SEM, với SEM = std / sqrt(N)).
  + Tỷ lệ tăng trưởng tương đối (Relative Gain %):
    Gain% = (Mean_Att - Mean_Baseline) / Mean_Baseline * 100%.
  + Paired Wilcoxon signed-rank test (two-sided, scipy.stats.wilcoxon) trên các lát cắt cặp đôi.
  + Đánh dấu mức ý nghĩa thống kê: *** (p < 0.001), ** (p < 0.01), * (p < 0.05), ns (p >= 0.05).
- Xuất file bảng tổng hợp khoa học cuối cùng: results/quantitative_comparison_table.csv.
- In bảng tổng quan định dạng Markdown chuẩn IEEE ra màn hình terminal.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

MODELS_ORDER = ["att_unet", "segnet", "unet"]
MODEL_DISPLAY = {
    "att_unet": "Attention U-Net (Proposed)",
    "segnet": "SegNet (Baseline)",
    "unet": "Standard U-Net (Baseline)",
}
METRICS = ["dice", "iou", "precision", "recall"]


def compute_mean_std_ci(values: List[float]) -> Tuple[float, float, float]:
    """Tính Mean, Standard Deviation, và 95% Confidence Interval (1.96 * SEM)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std_val / math.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem
    return mean_val, std_val, ci95


def perform_paired_wilcoxon(
    proposed_vals: List[float],
    baseline_vals: List[float],
) -> Tuple[float, float, str]:
    """Thực hiện Paired Wilcoxon signed-rank test giữa 2 mảng cặp đôi."""
    if wilcoxon is None or len(proposed_vals) < 5:
        return float("nan"), float("nan"), "N/A"

    diffs = np.array(proposed_vals) - np.array(baseline_vals)
    if np.all(np.abs(diffs) < 1e-12):
        return 0.0, 1.0, "ns"

    try:
        res = wilcoxon(proposed_vals, baseline_vals, zero_method="wilcox", alternative="two-sided")
        stat = float(res.statistic)
        p_val = float(res.pvalue)

        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        return stat, p_val, sig
    except Exception:
        return float("nan"), float("nan"), "err"


def load_slice_metrics(csv_path: Path) -> List[Dict[str, Any]]:
    """Đọc dữ liệu chi tiết từng lát cắt từ test_metrics_slice_level.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "model_key": r["model_key"],
                "sample_id": r["sample_id"],
                "patient_id": r["patient_id"],
                "slice_index": int(r["slice_index"]),
                "lesion_present": int(float(r["lesion_present"])),
                "tp": int(float(r.get("tp", 0))),
                "fp": int(float(r.get("fp", 0))),
                "fn": int(float(r.get("fn", 0))),
                "tn": int(float(r.get("tn", 0))),
                "dice": float(r["dice"]),
                "iou": float(r["iou"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
            })
    return rows


def analyze_scope(
    all_rows: List[Dict[str, Any]],
    scope_name: str,
    lesion_only: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Phân tích thống kê và kiểm định cho một Scope (Lesion Slices hoặc All Slices)."""
    if lesion_only:
        filtered_rows = [r for r in all_rows if r["lesion_present"] == 1]
    else:
        filtered_rows = all_rows

    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in filtered_rows:
        mkey = r["model_key"]
        sid = r["sample_id"]
        if mkey not in data:
            data[mkey] = {}
        data[mkey][sid] = {m: r[m] for m in METRICS}

    common_sids = sorted(
        set(data.get("att_unet", {}).keys())
        & set(data.get("unet", {}).keys())
        & set(data.get("segnet", {}).keys())
    )
    n_slices = len(common_sids)

    vectors: Dict[str, Dict[str, List[float]]] = {}
    stats_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for mkey in MODELS_ORDER:
        vectors[mkey] = {}
        stats_summary[mkey] = {}
        for metric in METRICS:
            vals = [data[mkey][sid][metric] for sid in common_sids]
            vectors[mkey][metric] = vals
            mean_v, std_v, ci_v = compute_mean_std_ci(vals)
            stats_summary[mkey][metric] = {
                "mean": mean_v,
                "std": std_v,
                "ci95": ci_v,
            }

    result_rows = []
    md_lines = [
        f"### {scope_name} (N = {n_slices} slices)",
        "",
        "| Model | DSC (Mean ± Std [95% CI]) | IoU (Mean ± Std) | Precision | Recall | Wilcoxon vs Att (DSC) | Wilcoxon vs Att (IoU) | Relative Gain (DSC) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    att_key = "att_unet"

    for mkey in MODELS_ORDER:
        model_name = MODEL_DISPLAY[mkey]
        row_dict: Dict[str, Any] = {
            "scope": scope_name,
            "model": model_name,
            "model_key": mkey,
            "n_slices": n_slices,
        }

        # Khởi tạo trước tất cả các trường so sánh
        for metric in METRICS:
            row_dict[f"gain_att_vs_{metric}_pct"] = ""
            row_dict[f"wilcoxon_stat_{metric}"] = ""
            row_dict[f"wilcoxon_p_{metric}"] = ""
            row_dict[f"wilcoxon_sig_{metric}"] = ""

        for metric in METRICS:
            m_stats = stats_summary[mkey][metric]
            row_dict[f"{metric}_mean"] = round(m_stats["mean"], 4)
            row_dict[f"{metric}_std"] = round(m_stats["std"], 4)
            row_dict[f"{metric}_ci95"] = round(m_stats["ci95"], 4)

            if mkey != att_key:
                base_mean = m_stats["mean"]
                att_mean = stats_summary[att_key][metric]["mean"]
                gain = ((att_mean - base_mean) / (base_mean + 1e-9)) * 100.0
                row_dict[f"gain_att_vs_{metric}_pct"] = round(gain, 2)

                stat_w, p_val, sig = perform_paired_wilcoxon(
                    vectors[att_key][metric],
                    vectors[mkey][metric],
                )
                row_dict[f"wilcoxon_stat_{metric}"] = round(stat_w, 2)
                row_dict[f"wilcoxon_p_{metric}"] = f"{p_val:.4e}"
                row_dict[f"wilcoxon_sig_{metric}"] = sig
            else:
                row_dict[f"gain_att_vs_{metric}_pct"] = "Reference"
                row_dict[f"wilcoxon_stat_{metric}"] = "ref"
                row_dict[f"wilcoxon_p_{metric}"] = "ref"
                row_dict[f"wilcoxon_sig_{metric}"] = "ref"

        result_rows.append(row_dict)

        dsc_str = f"{row_dict['dice_mean']:.4f} ± {row_dict['dice_std']:.4f} [±{row_dict['dice_ci95']:.4f}]"
        iou_str = f"{row_dict['iou_mean']:.4f} ± {row_dict['iou_std']:.4f}"
        prec_str = f"{row_dict['precision_mean']:.4f} ± {row_dict['precision_std']:.4f}"
        rec_str = f"{row_dict['recall_mean']:.4f} ± {row_dict['recall_std']:.4f}"

        if mkey == att_key:
            w_dsc = "Reference (Proposed)"
            w_iou = "Reference (Proposed)"
            gain_dsc = "-"
        else:
            w_dsc = f"p={float(row_dict['wilcoxon_p_dice']):.2e} ({row_dict['wilcoxon_sig_dice']})"
            w_iou = f"p={float(row_dict['wilcoxon_p_iou']):.2e} ({row_dict['wilcoxon_sig_iou']})"
            gain_dsc = f"{row_dict['gain_att_vs_dice_pct']:+.2f}%"

        bold = "**" if mkey == att_key else ""
        md_lines.append(
            f"| {bold}{model_name}{bold} | {bold}{dsc_str}{bold} | {bold}{iou_str}{bold} | {prec_str} | {rec_str} | {w_dsc} | {w_iou} | {gain_dsc} |"
        )

    md_table = "\n".join(md_lines)
    return result_rows, md_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EVAL-07: Statistical hypothesis testing and quantitative comparison"
    )
    parser.add_argument(
        "--input_slice_csv",
        type=Path,
        default=Path("results/test_metrics_slice_level.csv"),
        help="Đường dẫn file kết quả cấp lát cắt",
    )
    parser.add_argument(
        "--output_table_csv",
        type=Path,
        default=Path("results/quantitative_comparison_table.csv"),
        help="Đường dẫn file bảng so sánh định lượng cuối cùng",
    )
    args = parser.parse_args()

    print("======================================================================")
    print(" TASK EVAL-07: STATISTICAL ANALYSIS & QUANTITATIVE COMPARISON")
    print("======================================================================")

    all_rows = load_slice_metrics(args.input_slice_csv)
    print(f"[*] Loaded {len(all_rows)} total slice records from {args.input_slice_csv}.")

    # 1. Phân tích Scope 1: Primary Mode (Lesion Slices GT > 0, N=221)
    rows_primary, md_primary = analyze_scope(
        all_rows,
        scope_name="Primary Mode: Lesion-containing Slices (GT > 0)",
        lesion_only=True,
    )

    # 2. Phân tích Scope 2: Secondary Mode (All Slices, N=265)
    rows_secondary, md_secondary = analyze_scope(
        all_rows,
        scope_name="Secondary Mode: All Slices (Complete Test Set)",
        lesion_only=False,
    )

    combined_rows = rows_primary + rows_secondary

    # Thu thập fieldnames đầy đủ từ tất cả các rows
    fieldnames: List[str] = []
    for r in combined_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    args.output_table_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_table_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in combined_rows:
            filtered = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(filtered)

    print(f"[+] Exported final comparison table: {args.output_table_csv}")
    print("\n" + "=" * 90)
    print(" BẢNG TỔNG HỢP SO SÁNH ĐỊNH LƯỢNG KHOA HỌC (IEEE STANDARD REPORT)")
    print("=" * 90 + "\n")
    print(md_primary)
    print("\n" + "-" * 90 + "\n")
    print(md_secondary)
    print("\n" + "=" * 90)
    print("[SUCCESS] EVAL-07 statistical analysis completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
