# Báo cáo Tuần 5 — Đánh giá định lượng trên tập kiểm thử độc lập

**Tasks:** EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07
**Ngày chạy:** 2026-11-01
**Tập test:** `data/processed/test_manifest.json` — 265 lát cắt, 3 bệnh nhân
(221 lát có tổn thương, 44 lát lành), patient-level 70/15/15, seed 2026.

---

## 1. Tính hợp lệ của checkpoint (bắt buộc đọc trước khi dùng số liệu)

Các checkpoint cũ ngày 2025-10-25 **không dùng được** cho W5:

- Chúng được huấn luyện khi pipeline đọc mask nhị phân `{0, 1}` nhưng cấu hình
  dùng `lesion_label=3`, tạo ground truth rỗng. Vì vậy `val_dice = 1.0000`
  trong log cũ là điểm giả của trường hợp `GT = Pred = 0`.
- Kiểm tra trực tiếp: checkpoint Attention U-Net cũ không dự đoán pixel nào
  `> 0.5` trên toàn bộ test set.

Do đó 3 mô hình đã được **huấn luyện lại** trên pipeline đã sửa
(`lesion_label=1`). Cấu hình chung, giống nhau cho cả 3 mô hình:

| Thành phần | Giá trị |
|---|---|
| Input | 256 × 256, HU window [-1000, 400], min-max [0, 1] |
| Kiến trúc | Attention U-Net / U-Net / SegNet (config gốc của repo) |
| Loss | Hybrid 0.5·BCE + 0.5·Soft Dice |
| Optimizer | AdamW, lr = 1e-4, weight decay = 0.01 |
| Scheduler | CosineAnnealingLR, T_max = 30 |
| Batch size | 8 (AMP fp16, do GPU 4 GB) |
| Epochs | tối đa 30, early stopping patience = 15 |
| Augmentation | HorizontalFlip, ShiftScaleRotate, ElasticTransform (chỉ train) |
| Checkpoint chọn | epoch có val Dice cao nhất |

| Model | Epoch tốt nhất | Val Dice (slice-level) |
|---|---:|---:|
| Attention U-Net | 8 | 0.7015 |
| Standard U-Net | 10 | 0.7020 |
| SegNet | 24 | 0.6792 |

> Lưu ý: đây là lần huấn luyện lại hợp lệ đầu tiên; không có hyperparameter
> tuning trên tập test. Toàn bộ ngưỡng cố định 0.5, không hậu xử lý.

---

## 2. Quy ước tính metric

- **Cấp chính (báo cáo trong bài):** macro average trên từng lát cắt có
  tổn thương (`GT > 0`), N = 221.
- **Cấp bổ trợ:** (a) toàn bộ 265 lát; (b) patient-level, gộp TP/FP/FN của
  tất cả lát thuộc cùng bệnh nhân rồi mới tính metric.
- **Edge cases** đúng `Metrics_Specification.md`:
  - GT rỗng, Pred rỗng → DSC = IoU = Precision = Recall = 1.0
  - GT rỗng, Pred > 0 → DSC = IoU = Precision = 0.0; Recall = 1.0
  - GT > 0, Pred rỗng → tất cả bằng 0.0
- **95% CI:** xấp xỉ chuẩn `mean ± 1.96·std/√n`. Với patient-level,
  n = 3 nên CI rất rộng và chỉ mang tính mô tả.

---

## 3. Kết quả chính — lesion-containing slices (N = 221)

| Model | DSC | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Attention U-Net (Proposed) | 0.6312 ± 0.2648 | 0.5080 ± 0.2457 | 0.5854 ± 0.2775 | **0.8448 ± 0.2082** |
| Standard U-Net (Baseline) | **0.6837 ± 0.2364** | **0.5604 ± 0.2323** | 0.6349 ± 0.2576 | 0.8417 ± 0.2198 |
| SegNet (Baseline) | 0.6693 ± 0.2407 | 0.5425 ± 0.2228 | **0.6466 ± 0.2406** | 0.7557 ± 0.2849 |

95% CI (DSC): Attention U-Net [0.5963, 0.6662]; U-Net [0.6525, 0.7149];
SegNet [0.6376, 0.7011]. Chi tiết đầy đủ trong
`results/metrics_dsc_summary.csv` (và tương tự cho IoU/Precision/Recall).

## 4. Kết quả bổ trợ — patient-level (N = 3 bệnh nhân)

| Model | DSC | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Attention U-Net | 0.5825 ± 0.3417 | 0.4607 ± 0.3098 | 0.5105 ± 0.3519 | 0.8969 ± 0.0721 |
| Standard U-Net | 0.6890 ± 0.1922 | 0.5461 ± 0.2110 | 0.5924 ± 0.2457 | **0.9058 ± 0.0597** |
| SegNet | **0.7256 ± 0.0924** | **0.5747 ± 0.1112** | **0.6616 ± 0.1062** | 0.8064 ± 0.0769 |

DSC từng bệnh nhân:

| Patient | Attention U-Net | U-Net | SegNet |
|---|---:|---:|---:|
| P_coronacases_002 | 0.7971 | 0.8271 | 0.8026 |
| P_coronacases_007 | 0.7620 | 0.7703 | 0.7510 |
| P_radiopaedia_27_86410_0 | 0.1884 | 0.4695 | 0.6232 |

## 5. Kết quả bổ trợ — toàn bộ 265 lát (gồm 44 lát lành)

| Model | DSC | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Attention U-Net | 0.6170 ± 0.3175 | 0.5142 ± 0.3030 | 0.5788 ± 0.3251 | **0.8706 ± 0.1987** |
| Standard U-Net | **0.6683 ± 0.2968** | **0.5655 ± 0.2922** | 0.6276 ± 0.3096 | 0.8679 ± 0.2091 |
| SegNet | 0.6488 ± 0.3029 | 0.5430 ± 0.2875 | **0.6298 ± 0.3016** | 0.7963 ± 0.2756 |

---

## 6. Kiểm định thống kê (EVAL-07)

Paired test trên từng lát cắt (N = 221 cặp), so sánh Attention U-Net với
từng baseline. Bảng đầy đủ: `results/quantitative_comparison_table.csv`.

| So sánh | Metric | Wilcoxon p | t-test p | Kết luận (α = 0.05) |
|---|---|---:|---:|---|
| vs U-Net | DSC | 1.00e-14 | 2.06e-08 | Có ý nghĩa — U-Net tốt hơn |
| vs U-Net | IoU | 4.63e-15 | 1.36e-09 | Có ý nghĩa — U-Net tốt hơn |
| vs U-Net | Precision | 7.41e-15 | 8.15e-08 | Có ý nghĩa — U-Net tốt hơn |
| vs U-Net | Recall | 0.273 | 0.771 | Không có ý nghĩa |
| vs SegNet | DSC | 0.0184 | 0.00365 | Có ý nghĩa — SegNet tốt hơn |
| vs SegNet | IoU | 0.0117 | 0.00218 | Có ý nghĩa — SegNet tốt hơn |
| vs SegNet | Precision | 6.72e-05 | 1.28e-05 | Có ý nghĩa — SegNet tốt hơn |
| vs SegNet | Recall | 4.61e-09 | 1.16e-08 | Có ý nghĩa — Attention U-Net tốt hơn |

Tóm tắt: Attention U-Net có Recall cao nhất (chênh U-Net không đáng kể,
chênh SegNet rõ rệt), nhưng thấp hơn có ý nghĩa ở DSC/IoU/Precision so với
cả hai baseline.

---

## 7. Thảo luận & cảnh báo

1. **Không có bằng chứng Attention U-Net vượt baseline trên bộ dữ liệu này.**
   Kết quả nghiêng về U-Net (DSC/IoU/Precision) và SegNet (DSC/IoU/Precision,
   patient-level), trong khi Attention U-Net chỉ nhỉnh Recall. Đây là kết quả
   thực nghiệm, không phải lỗi tính toán; cần báo cáo trung thực.
2. **Lỗi over-segmentation của Attention U-Net** tập trung ở bệnh nhân
   `P_radiopaedia_27_86410_0` (DSC 0.1884): mô hình phủ gần như toàn bộ nhu mô
   phổi thay vì vùng tổn thương. Đây là domain shift rõ rệt (ảnh 630×630,
   khác scanner) và là nguyên nhân chính kéo DSC trung bình xuống.
3. **Macro vs pooled:** DSC macro của Attention U-Net là 0.631 nhưng DSC gộp
   toàn bộ pixel chỉ 0.389, do một số lát bị FP rất lớn. Báo cáo chính dùng
   macro theo đúng đặc tả, nhưng nên nêu pooled/micro trong phần hạn chế để
   tránh hiểu nhầm.
4. **Cỡ mẫu nhỏ:** chỉ 3 bệnh nhân test; CI patient-level rất rộng. Kiểm định
   theo lát cắt coi các lát là độc lập (không phản ánh đầy đủ cấu trúc bệnh
   nhân). Cần thêm bệnh nhân hoặc bootstrap theo cụm bệnh nhân ở các tuần sau.
5. **Một seed, không tune:** kết quả có thể thay đổi với seed/khởi tạo khác.
   Không được dùng kết quả này để chọn hyperparameter trên test.

---

## 8. Sản phẩm bàn giao

| Task | File |
|---|---|
| EVAL-03 | `results/metrics_dsc_detail.csv`, `results/metrics_dsc_patient.csv`, `results/metrics_dsc_summary.csv` |
| EVAL-04 | `results/metrics_iou_detail.csv`, `results/metrics_iou_patient.csv`, `results/metrics_iou_summary.csv` |
| EVAL-05 | `results/metrics_precision_detail.csv`, `results/metrics_precision_patient.csv`, `results/metrics_precision_summary.csv` |
| EVAL-06 | `results/metrics_recall_detail.csv`, `results/metrics_recall_patient.csv`, `results/metrics_recall_summary.csv` |
| EVAL-07 | `results/quantitative_comparison_table.csv`, `results/quantitative_comparison_table_all_slices.csv`, `results/quantitative_comparison_table.tex` |

Dự đoán gốc (probability map, threshold 0.5):
`predictions/att_unet/`, `predictions/unet/`, `predictions/segnet/`
(mỗi thư mục có `predictions_manifest.json` ghi checkpoint và cấu hình).

## 9. Lệnh tái lập

```bash
# 1) Cache 2D slices (một lần)
python scripts/build_slice_cache.py

# 2) Huấn luyện lại 3 mô hình (cùng protocol)
python train.py --config configs/attention_unet.yaml \
  --cache_dir data/processed/slice_cache --epochs 30 --batch_size 8 --amp
python train.py --config configs/baseline_unet.yaml \
  --cache_dir data/processed/slice_cache --epochs 30 --batch_size 8 --amp
python train.py --config configs/segnet.yaml \
  --cache_dir data/processed/slice_cache --epochs 30 --batch_size 8 --amp

# 3) Export dự đoán
python scripts/export_predictions.py --checkpoint checkpoints/EXP-ATTUNET-001_best.pth \
  --model AttentionUNet --output_dir predictions/att_unet
python scripts/export_predictions.py --checkpoint checkpoints/EXP-UNET-001_best.pth \
  --model UNet --output_dir predictions/unet
python scripts/export_predictions.py --checkpoint checkpoints/EXP-SEGNET-001_best.pth \
  --model SegNet --output_dir predictions/segnet

# 4) Tính metric + kiểm định
python scripts/compute_dsc.py --predictions_dir predictions
python scripts/compute_iou.py --predictions_dir predictions
python scripts/compute_precision.py --predictions_dir predictions
python scripts/compute_recall.py --predictions_dir predictions
python scripts/compute_wilcoxon.py
```
