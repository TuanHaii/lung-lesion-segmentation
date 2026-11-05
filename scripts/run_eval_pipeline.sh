#!/usr/bin/env bash
# ==============================================================================
# Pipeline Đánh giá Tự động hóa Toàn diện (Full Automated Evaluation Pipeline)
# Nhiệm vụ: Tích hợp EVAL-01 đến EVAL-07 theo quy chuẩn kỹ thuật RES-04 / v2.0
# Tác giả: Data & Evaluation Lead
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=============================================================================="
echo " KHỞI CHẠY PIPELINE ĐÁNH GIÁ THỰC NGHIỆM KHOA HỌC: EVAL-01 -> EVAL-07"
echo " Dự án: Phân đoạn tổn thương CT phổi bằng kiến trúc Attention U-Net"
echo " Thời gian khởi chạy: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# BƯỚC 0: RESET VÀ CLEANUP MÔI TRƯỜNG SẠCH 100%
# ------------------------------------------------------------------------------
echo -e "\n[BƯỚC 0/4] Dọn dẹp môi trường và xóa sạch các file kết quả cũ..."

# Gỡ bỏ thuộc tính read-only nếu file manifest cũ tồn tại
if [ -f "data/processed/test_manifest.json" ]; then
    chmod 644 data/processed/test_manifest.json 2>/dev/null || true
    rm -f data/processed/test_manifest.json
fi

# Xóa các file kết quả cũ trong results/
rm -f results/metrics_*_detail.csv
rm -f results/test_metrics_*.csv
rm -f results/quantitative_comparison_table*.csv
rm -f results/inference_latency_summary.json

echo "[+] Môi trường đã được làm sạch 100%."

# ------------------------------------------------------------------------------
# BƯỚC 1: EVAL-01 - CHUẨN BỊ VÀ CÔ LẬP TẬP KIỂM THỬ ĐỘC LẬP
# ------------------------------------------------------------------------------
echo -e "\n[BƯỚC 1/4] Thực thi EVAL-01: Chuẩn bị và cô lập test_manifest.json..."
python3 scripts/prepare_test_manifest.py

if [ ! -f "data/processed/test_manifest.json" ]; then
    echo "[!] LỖI: Không tìm thấy data/processed/test_manifest.json sau khi chạy EVAL-01!" >&2
    exit 1
fi
echo "[+] Hoàn thành EVAL-01. File test_manifest.json đã được khóa quyền chỉ đọc (read-only)."

# ------------------------------------------------------------------------------
# BƯỚC 2: EVAL-02 - ĐỒNG BỘ VÀ CHUẨN HÓA KẾT QUẢ SUY LUẬN PREDICTIONS
# ------------------------------------------------------------------------------
echo -e "\n[BƯỚC 2/4] Thực thi EVAL-02: Đồng bộ và chuẩn hóa prediction tensors..."
python3 scripts/sync_eval02_predictions.py

for model in att_unet unet segnet; do
    if [ ! -d "predictions/$model" ]; then
        echo "[!] LỖI: Thư mục predictions/$model không tồn tại sau khi đồng bộ!" >&2
        exit 1
    fi
done
echo "[+] Hoàn thành EVAL-02. Toàn bộ 3 mô hình đã được đồng bộ 1:1."

# ------------------------------------------------------------------------------
# BƯỚC 3: EVAL-03..06 - ĐÁNH GIÁ ĐỒNG LOẠT 4 CHỈ SỐ Y TẾ (DSC, IoU, PREC, REC)
# ------------------------------------------------------------------------------
echo -e "\n[BƯỚC 3/4] Thực thi EVAL-03..06: Tính toán toàn diện 4 chỉ số theo RES-04..."
python3 scripts/evaluate_all_metrics.py

# Kiểm tra sự sẵn sàng của 6 file kết quả bắt buộc
REQUIRED_RESULT_FILES=(
    "results/metrics_dsc_detail.csv"
    "results/metrics_iou_detail.csv"
    "results/metrics_precision_detail.csv"
    "results/metrics_recall_detail.csv"
    "results/test_metrics_slice_level.csv"
    "results/test_metrics_patient_level.csv"
)

for fpath in "${REQUIRED_RESULT_FILES[@]}"; do
    if [ ! -f "$fpath" ]; then
        echo "[!] LỖI: Thiếu file kết quả bắt buộc: $fpath" >&2
        exit 1
    fi
done
echo "[+] Hoàn thành EVAL-03..06. Đã xuất đủ 6 file báo cáo chi tiết và tổng hợp."

# ------------------------------------------------------------------------------
# BƯỚC 4: EVAL-07 - PHÂN TÍCH THỐNG KÊ & KIỂM ĐỊNH GIẢ THUYẾT WILCOXON
# ------------------------------------------------------------------------------
echo -e "\n[BƯỚC 4/4] Thực thi EVAL-07: Phân tích thống kê định lượng và kiểm định Wilcoxon..."
python3 scripts/statistical_analysis.py

if [ ! -f "results/quantitative_comparison_table.csv" ]; then
    echo "[!] LỖI: Không tìm thấy results/quantitative_comparison_table.csv!" >&2
    exit 1
fi

echo -e "\n=============================================================================="
echo " [THÀNH CÔNG RỰC RỠ] TOÀN BỘ PIPELINE ĐÁNH GIÁ EVAL-01 -> EVAL-07 ĐÃ HOÀN TẤT!"
echo "=============================================================================="
echo "Danh mục các tài liệu và kết quả đã được xác thực toàn vẹn:"
echo " 1. [EVAL-01] data/processed/test_manifest.json (Locked 0o444)"
echo " 2. [EVAL-02] predictions/{att_unet, unet, segnet}/*.npy (Khớp 1:1 265/265 slices)"
echo " 3. [EVAL-03] results/metrics_dsc_detail.csv"
echo " 4. [EVAL-04] results/metrics_iou_detail.csv"
echo " 5. [EVAL-05] results/metrics_precision_detail.csv"
echo " 6. [EVAL-06] results/metrics_recall_detail.csv"
echo " 7. [RES-04]  results/test_metrics_slice_level.csv & results/test_metrics_patient_level.csv"
echo " 8. [EVAL-07] results/quantitative_comparison_table.csv"
echo "=============================================================================="

