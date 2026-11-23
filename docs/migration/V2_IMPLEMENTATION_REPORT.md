# V2 IMPLEMENTATION REPORT
## Lung Lesion Segmentation | Dual-Source Dataset Architecture

- **Date:** 2026-11-23 (Asia/Saigon)
- **Author:** Senior ML Engineer & Research Software Engineer
- **Git Branch:** `main`
- **Git HEAD Commit:** `d6aad6cbaf8fe6bc4de3ca5ae972be5c91c41b0c`
- **Document:** `docs/migration/V2_IMPLEMENTATION_REPORT.md`

---

## 1. Overview of Changes

Pipeline V2 đã được thiết kế và triển khai thành công nhằm giải quyết triệt để 5 vấn đề cốt lõi của V1:
1. **Lỗi Resolver Fallback:** Thay thế logic fallback nguy hiểm bằng `SafePathResolver` có cơ chế cô lập từ khóa loại file nghiêm ngặt, bắt buộc CT và Mask không trùng lặp, và ném lỗi rõ ràng khi thiếu file.
2. **Khuyết tật Preprocessing 2 nguồn:** Tách biệt thành hai profile chuyên biệt:
   - `CoronacasesPreprocessor`: HU window `[-1000, 400]`, chuẩn hóa tuyến tính `[0, 1]`.
   - `RadiopaediaPreprocessor`: Chuẩn hóa trực tiếp `uint8 [0, 255] -> [0.0, 1.0]`, bảo toàn độ tương phản gốc, không áp dụng HU window.
3. **Thống nhất Training Engine:** Cả ba mô hình (U-Net, Attention U-Net, SegNet) sử dụng chung `train.py`. Đánh dấu `train_segnet.py` là deprecated.
4. **Sửa lỗi Metric Aggregation:** Cung cấp `compute_sample_metrics` tính toán Macro per-sample metrics cho validation.
5. **Độc lập và an toàn trong Evaluation:** Triển khai `EvaluatorV2` đa tầng (Overall micro, Macro all/lesion/normal slices, Patient-level, Source-level), loại bỏ hoàn toàn cơ chế tạo mock data khi thiếu ground truth.

---

## 2. File Implementation Summary

| File / Thư mục | Hành động | Mục đích | Trạng thái |
|---|---|---|---|
| `docs/migration/V2_BASELINE_AUDIT.md` | Create | Báo cáo chi tiết audit hiện trạng V1, bằng chứng lỗi resolver và preprocessing | Hoàn thành |
| `docs/migration/V2_ARCHITECTURE.md` | Create | Bản thiết kế kiến trúc kỹ thuật mục tiêu Pipeline V2 | Hoàn thành |
| `docs/migration/V2_MIGRATION_PLAN.md` | Create | Kế hoạch các bước chuyển đổi và tiêu chuẩn an toàn | Hoàn thành |
| `src/data/v2/source_detector.py` | Create | Module nhận diện nguồn (`coronacases` vs `radiopaedia`) nghiêm ngặt | Hoàn thành |
| `src/data/v2/resolver.py` | Create | `SafePathResolver` kiểm soát type-safety, không fallback bừa bãi, cấm trùng file | Hoàn thành |
| `src/data/v2/preprocessing/base.py` | Create | Interface trừu tượng `BasePreprocessor` | Hoàn thành |
| `src/data/v2/preprocessing/coronacases.py` | Create | Preprocessor cho Coronacases (HU clipping `[-1000, 400]` và scaling `[0, 1]`) | Hoàn thành |
| `src/data/v2/preprocessing/radiopaedia.py` | Create | Preprocessor cho Radiopaedia (Direct scaling `[0, 255] -> [0.0, 1.0]`) | Hoàn thành |
| `src/data/v2/preprocessing/common.py` | Create | Xử lý không gian đồng bộ (resize bilinear ảnh, nearest mask, nhị phân hóa) | Hoàn thành |
| `src/data/v2/preprocessing/registry.py` | Create | Registry quản lý và phân phối preprocessor theo nguồn | Hoàn thành |
| `src/data/v2/preprocessing/__init__.py` | Create | Package init cho preprocessing V2 | Hoàn thành |
| `src/data/v2/manifest.py` | Create | Công cụ tạo và nạp Manifest V2 (bảo toàn V1 manifest) | Hoàn thành |
| `src/data/v2/integrity.py` | Create | Data Integrity Validator độc lập kiểm tra split leakage, nhãn, phân bổ | Hoàn thành |
| `src/data/v2/dataset.py` | Create | `LungCTDatasetV2` và `get_dataloader_v2` tích hợp toàn bộ luồng V2 | Hoàn thành |
| `src/data/v2/__init__.py` | Create | Package init cho Data V2 | Hoàn thành |
| `data/processed_v2/clean_slices_manifest_v2.csv` | Create | Manifest V2 chứa 1,750 lát cắt với metadata nguồn và split | Hoàn thành |
| `src/evaluation/v2_evaluator.py` | Create | Engine đánh giá độc lập đa tầng không mock fallback | Hoàn thành |
| `src/evaluation/__init__.py` | Create | Package init cho Evaluation V2 | Hoàn thành |
| `src/utils/metrics.py` | Modify | Thêm `compute_sample_metrics` tính toán macro per-sample chuẩn | Hoàn thành |
| `train.py` | Modify | Hỗ trợ V2 DataLoader, macro validation metrics, và Checkpoint V2 metadata | Hoàn thành |
| `train_segnet.py` | Modify | Thêm cảnh báo deprecation, khuyến nghị sử dụng `train.py --config configs/segnet.yaml` | Hoàn thành |
| `tests/test_pipeline_v2.py` | Create | Bộ test toàn diện 20 unit/integration tests kiểm thử mọi khía cạnh V2 | Hoàn thành |
| `docs/migration/V2_VALIDATION_REPORT.md` | Create | Báo cáo kiểm định và kết quả thử nghiệm kỹ thuật | Hoàn thành |

---

## 3. Git Working Tree State

- **Branch hiện tại:** `main`
- **HEAD Commit:** `d6aad6cbaf8fe6bc4de3ca5ae972be5c91c41b0c`
- **File V1 được bảo toàn:**
  - `data/processed/clean_slices_manifest.csv` (Giữ nguyên)
  - `data/processed/splits_patient_level.json` (Giữ nguyên)
  - `data/processed/test_manifest.json` (Giữ nguyên)
  - `data/processed/summary_split.csv` (Giữ nguyên)
  - Tất cả các checkpoints cũ trong `checkpoints/` (Giữ nguyên)
  - Tất cả predictions và reports trong `reports/`, `results/`, `predictions/` (Giữ nguyên)
- **Tình trạng working tree:** Sạch, không có bất kỳ lệnh revert, reset hay mất mát dữ liệu nào xảy ra.
