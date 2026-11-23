# V2 MIGRATION PLAN
## Step-by-Step Implementation and Verification Roadmap

- **Date:** 2026-11-23
- **Author:** Senior ML Engineer & Research Software Engineer
- **Document:** `docs/migration/V2_MIGRATION_PLAN.md`

---

## 1. Migration Strategy Overview

Quá trình chuyển đổi từ V1 sang V2 tuân thủ các nguyên tắc an toàn tuyệt đối:
- **Zero In-place Overwrites:** Toàn bộ code mới được viết trong các module chuyên biệt (`src/data/v2/`, `src/evaluation/v2_evaluator.py`, `data/processed_v2/`). Các file V1 gốc được giữ nguyên làm baseline đối chứng.
- **Safety First:** Không thực hiện bất kỳ lệnh training nào, không cập nhật trọng số, không ghi đè checkpoint.
- **Evidence-Based:** Mọi quyết định thiết kế và preprocessing đều có kiểm chứng bằng unit test và phân tích số học.

---

## 2. Implementation Phases & Milestones

### Phase 0: Repository Discovery & Baseline Audit (HOÀN THÀNH)
- [x] Kiểm tra Git state, HEAD commit, modified và untracked files.
- [x] Đọc các tài liệu dự án (`01_PROJECT_CHARTER.md`, `04_TECH_STACK_AND_DATASET.md`, `SEGNET_ROOT_CAUSE_ANALYSIS.md`).
- [x] Tái lập và xác minh lỗi Resolver V1 (CT và Mask cùng trỏ về `coronacases_001.nii`).
- [x] Phân tích khuyết tật tiền xử lý HU windowing trên dữ liệu `uint8` Radiopaedia.
- [x] So sánh `train.py` và `train_segnet.py`.
- [x] Viết báo cáo `docs/migration/V2_BASELINE_AUDIT.md`.

### Phase 1: Architecture Design (HOÀN THÀNH)
- [x] Thiết kế end-to-end data flow cho dual-source preprocessing.
- [x] Thiết kế module phân loại nguồn `SourceDetector`.
- [x] Thiết kế `SafePathResolver` chống fallback sai loại file.
- [x] Thiết kế cấu trúc tiền xử lý riêng `CoronacasesPreprocessor` và `RadiopaediaPreprocessor`.
- [x] Thiết kế cấu trúc Manifest V2 và Data Integrity Validator.
- [x] Thiết kế cấu hình thống nhất và Checkpoint metadata schema cho cả ba mô hình.
- [x] Viết tài liệu `docs/migration/V2_ARCHITECTURE.md`.
- [x] Viết tài liệu `docs/migration/V2_MIGRATION_PLAN.md`.

### Phase 2: Implement Data V2 Modules (TIẾP THEO)
Triển khai mã nguồn trong `src/data/v2/`:
1. `src/data/v2/__init__.py`
2. `src/data/v2/source_detector.py`: Phân loại nguồn `coronacases` vs `radiopaedia` nghiêm ngặt.
3. `src/data/v2/resolver.py`: `SafePathResolver` với kiểm tra chặt chẽ type-keyword, existence, và non-identity.
4. `src/data/v2/preprocessing/base.py`: Interface trừu tượng cho tiền xử lý ảnh.
5. `src/data/v2/preprocessing/coronacases.py`: Preprocessing Coronacases (HU clipping `[-1000, 400]`, scaling `[0, 1]`).
6. `src/data/v2/preprocessing/radiopaedia.py`: Preprocessing Radiopaedia (scaling `[0, 255] -> [0.0, 1.0]`, bảo toàn tương phản).
7. `src/data/v2/preprocessing/common.py`: Spatial transforms (resize bilinear cho ảnh, nearest cho mask, nhị phân hóa nhãn).
8. `src/data/v2/preprocessing/registry.py`: Registry quản lý preprocessors.
9. `src/data/v2/manifest.py`: Trình đọc và tạo Manifest V2, trích xuất metadata và gán preprocessing profile.
10. `src/data/v2/integrity.py`: Bộ kiểm tra toàn vẹn dữ liệu (Data Integrity Validator) độc lập.
11. `src/data/v2/dataset.py`: `LungCTDatasetV2` tích hợp đầy đủ các thành phần trên, trả về tensor chuẩn `[1, 256, 256]`.

### Phase 3: Unified Training Pipeline
1. Cập nhật `train.py` để hỗ trợ cờ chọn Dataset V2 (`--use_v2_data` hoặc tự động chọn khi cấu hình chỉ định V2 manifest).
2. Sửa lỗi metric aggregation trong `src/utils/metrics.py` hoặc tạo `compute_sample_metrics` để tính toán chính xác macro validation metrics per-sample.
3. Cập nhật checkpoint saving schema chuẩn V2 với đầy đủ metadata: git commit, manifest sha, split sha, config, preprocessing version, seed.
4. Đánh dấu `train_segnet.py` là deprecated, chuyển cấu hình SegNet chuẩn sang `configs/segnet.yaml` dùng chung `train.py`.

### Phase 4: Unified Evaluation Pipeline V2
1. Triển khai `src/evaluation/v2_evaluator.py`:
   - Tính toán đầy đủ: Overall Micro Metrics, Macro Slice Metrics, Patient-level Metrics, Source-level Metrics.
   - Phân tích chi tiết: All slices, Lesion-positive slices, Lesion-negative slices.
   - Nghiêm cấm fallback tạo dữ liệu mock; nếu thiếu dữ liệu, raise exception rõ ràng.

### Phase 5: Verification & Testing
1. **Unit Tests:**
   - Test Resolver: đúng CT, đúng Mask, phát hiện trùng file, phát hiện thiếu file, chặn fallback sai loại.
   - Test Preprocessing: Coronacases (HU clipping), Radiopaedia (255 scaling), output shapes, dải `[0, 1]`, dtype float32, không NaN/Inf.
   - Test Data Integrity: kiểm tra manifest V2, kiểm tra split không rò rỉ.
   - Test Evaluation Metrics: kiểm tra số học TP/FP/FN, empty mask handling, micro vs macro.
2. **DataLoader Tests:**
   - Kiểm tra shape, dtype, batch loading trên DataLoader V2 (với dữ liệu test/synthetic nếu thiếu raw NIfTI).
3. **Model Forward Tests:**
   - Khởi tạo U-Net, Attention U-Net, SegNet và chạy forward pass trên tensor giả lập `[2, 1, 256, 256]`.
   - Xác nhận output shape `[2, 1, 256, 256]`. Tuyệt đối không gọi backward hay optimizer.step.

### Phase 6: Documentation & Reporting
1. Viết `docs/migration/V2_IMPLEMENTATION_REPORT.md`: Chi tiết tất cả file đã tạo/sửa, mục đích và trạng thái.
2. Viết `docs/migration/V2_VALIDATION_REPORT.md`: Báo cáo kết quả kiểm thử, tính toàn vẹn và mức độ sẵn sàng cho training.
3. Tổng hợp báo cáo kết luận gửi người dùng.
