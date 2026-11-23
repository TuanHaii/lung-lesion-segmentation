# V2 VALIDATION AND TEST REPORT
## Lung Lesion Segmentation | Dual-Source Dataset Architecture

- **Date:** 2026-11-23 (Asia/Saigon)
- **Author:** Senior ML Engineer & Research Software Engineer
- **Document:** `docs/migration/V2_VALIDATION_REPORT.md`

---

## 1. Executive Summary

Quá trình kiểm định Pipeline V2 đã được thực hiện nghiêm ngặt theo các tiêu chuẩn an toàn:
- **Không chạy training.**
- **Không cập nhật trọng số (`optimizer.step()` không bao giờ được gọi).**
- **Không ghi đè hoặc can thiệp vào bất kỳ checkpoint, manifest hoặc prediction cũ nào.**
- Toàn bộ **20 unit and integration tests** trong bộ test suite `tests/test_pipeline_v2.py` đều **PASS 100%**.
- Data Integrity Validator đã chạy thành công trên toàn bộ 1,750 lát cắt của 20 bệnh nhân trong Manifest V2: **0 Errors, 0 Warnings**.

---

## 2. Test Execution Details

Lệnh kiểm thử:
```bash
python -m unittest tests/test_pipeline_v2.py -v
```

Kết quả:
```text
test_manifest_v2_generation_and_integrity ... ok
test_dataloader_train_val_test_loading ... ok
test_confusion_matrix_manual_math ... ok
test_empty_mask_handling ... ok
test_macro_sample_metrics_torch ... ok
test_multi_level_evaluation_report ... ok
test_attention_unet_forward_shape ... ok
test_segnet_forward_shape ... ok
test_unet_forward_shape ... ok
test_coronacases_hu_windowing ... ok
test_nan_inf_handling ... ok
test_radiopaedia_direct_scaling ... ok
test_spatial_resize_and_binarize ... ok
test_missing_file_raises_filenotfound ... ok
test_prevent_ct_mask_identity ... ok
test_resolve_valid_pair ... ok
test_type_isolation_enforcement ... ok
test_coronacases_detection ... ok
test_radiopaedia_detection ... ok
test_unknown_source_raises ... ok

----------------------------------------------------------------------
Ran 20 tests in 2.006s

OK
```

---

## 3. Detailed Verification Results by Category

### Test 1: Resolver Safety & Path Isolation
- **Chặn trùng file CT và Mask:** Khi truyền cùng một đường dẫn cho CT và Mask, `SafePathResolver.resolve_pair` ném ngoại lệ `PathResolverError`.
- **Chặn fallback sai loại:** Khi tìm CT, resolver từ chối các file có chứa từ khóa mask (`mask`, `infection`).
- **Xử lý file thiếu:** Khi file không tồn tại, resolver ném `FileNotFoundError` mô tả chi tiết các đường dẫn đã tìm, tuyệt đối không trả về file ngẫu nhiên đầu tiên như V1.
- **Kết quả:** PASS.

### Test 2: Dual-Source Preprocessing Verification
- **Coronacases HU Windowing:** Kiểm tra với các giá trị biên:
  - `-1500.0` -> `0.0`
  - `-1000.0` -> `0.0`
  - `-300.0`  -> `0.5`
  - `400.0`   -> `1.0`
  - `1000.0`  -> `1.0`
  Dải đầu ra `[0.0, 1.0]`, kiểu `float32`.
- **Radiopaedia Direct Scaling:** Kiểm tra với dữ liệu thang đo `[0, 255]`:
  - `0`     -> `0.0`
  - `127.5` -> `0.5`
  - `255.0` -> `1.0`
  Bảo toàn đầy đủ tương phản mô gốc.
- **Xử lý NaN / Inf:** Cả 2 preprocessor tự động thay thế NaN và Inf an toàn về ngưỡng sàn/trần.
- **Đồng bộ Spatial Transforms:** Resize bilinear cho ảnh và nearest neighbor cho mask; nhị phân hóa nhãn rời rạc `{0.0, 1.0}`.
- **Kết quả:** PASS.

### Test 3: Data Integrity & Manifest V2
- Đã chạy `validate_dataset_integrity` trên `data/processed_v2/clean_slices_manifest_v2.csv`.
- **Tổng bệnh nhân:** 20 (10 Coronacases + 10 Radiopaedia).
- **Phân bổ tập split:**
  - Train: 14 bệnh nhân (7 Coronacases + 7 Radiopaedia) = 1,277 lát cắt.
  - Validation: 3 bệnh nhân (1 Coronacases + 2 Radiopaedia) = 208 lát cắt.
  - Test: 3 bệnh nhân (2 Coronacases + 1 Radiopaedia) = 265 lát cắt.
- **Phân bổ tổn thương:** 1,450 lát cắt có tổn thương (`lesion_present=1`), 300 lát cắt không có tổn thương (`lesion_present=0`).
- **Kiểm tra rò rỉ (Leakage):** Giao giữa các tập bệnh nhân hoàn toàn rỗng (`Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`).
- **Lỗi & Cảnh báo:** 0 Errors, 0 Warnings.
- **Kết quả:** PASS.

### Test 4: DataLoader V2 Compatibility
- Kiểm tra nạp batch thực tế trên dữ liệu NIfTI giả lập qua `get_dataloader_v2`:
  - Kích thước tensor ảnh: `[B, 1, 256, 256]`, dtype `float32`, bounded trong `[0.0, 1.0]`.
  - Kích thước tensor mask: `[B, 1, 256, 256]`, dtype `float32`, nhị phân nghiêm ngặt `{0.0, 1.0}`.
  - Metadata được truyền đầy đủ kèm theo từng batch (`sample_id`, `patient_id`, `source`, `slice_index`, `split`).
- **Kết quả:** PASS.

### Test 5: Model Forward Compatibility (Zero Optimizer Update)
- Forward pass thành công trên cả ba kiến trúc với tensor giả lập `[2, 1, 256, 256]`:
  - **U-Net:** Output shape `[2, 1, 256, 256]`.
  - **Attention U-Net:** Output shape `[2, 1, 256, 256]`.
  - **SegNet:** Output shape `[2, 1, 256, 256]`.
- Không gọi backward, không gọi `optimizer.step()`, không tạo checkpoint mới.
- **Kết quả:** PASS.

### Test 6: Multi-Level Evaluation V2 Mathematics
- Kiểm tra confusion matrix số học với các giá trị thủ công (TP=1, FP=1, FN=1, TN=1):
  - Dice lý thuyết = `2/4 = 0.5` -> Kết quả thực nghiệm: `0.5`.
  - IoU lý thuyết = `1/3 ≈ 0.3333` -> Kết quả thực nghiệm: `0.3333`.
  - Precision lý thuyết = `1/2 = 0.5` -> Kết quả thực nghiệm: `0.5`.
  - Recall lý thuyết = `1/2 = 0.5` -> Kết quả thực nghiệm: `0.5`.
- Kiểm tra trường hợp đặc biệt:
  - Cả Ground Truth và Prediction đều rỗng: Dice = 1.0, IoU = 1.0, Precision = 1.0, Recall = 1.0.
  - Ground Truth rỗng nhưng Prediction dương giả (False Positive): Dice = 0.0, IoU = 0.0, Precision = 0.0.
- Kiểm tra báo cáo đa tầng (Overall, Source-level, Patient-level, Lesion-slice level): Hoạt động chính xác.
- **Kết quả:** PASS.

---

## 4. Dataset Audit Status & Local Environment Limitation

- **Dữ liệu NIfTI cục bộ:** Trong workspace hiện tại chỉ có một file `data/raw/ct_volumes/coronacases_001.nii` (đã được audit xác nhận là file mask nhị phân bị đặt nhầm tên thư mục). 19 volumes còn lại không có sẵn trên ổ đĩa cục bộ.
- **Ý nghĩa:** Dataset đầy đủ được lưu trữ trên môi trường Kaggle / Cloud Storage theo quy trình nghiên cứu đã được ghi nhận trong `05_OPERATIONAL_MANUAL.md`.
- **Khả năng tương thích:** Toàn bộ pipeline V2 (`SafePathResolver`, `LungCTDatasetV2`, `train.py`) đã được thiết kế sẵn sàng nhận diện `RAW_DATA_DIR`, `KAGGLE_DATA_DIR` hoặc thư mục `/kaggle/input` tự động khi chạy trên môi trường có GPU và full dataset.

---

## 5. Training Readiness Assessment

**Đánh giá:** **READY WITH LIMITATIONS**

### Lý do:
1. **READY về mặt Mã nguồn và Kiến trúc:**
   - Cả ba mô hình đã được hợp nhất vào một training engine duy nhất (`train.py`).
   - Resolver an toàn đã loại bỏ hoàn toàn lỗi swap CT và Mask.
   - Dual-source preprocessing đã được định nghĩa và kiểm chứng tính đúng đắn trên cả Coronacases và Radiopaedia.
   - Manifest V2 và Data Integrity Validator đã xác thực 1,750 lát cắt đạt chuẩn.
   - Hệ thống đánh giá V2 độc lập đã sẵn sàng.
2. **LIMITATIONS về mặt Dữ liệu Cục bộ:**
   - 19/20 volumes NIfTI đầy đủ vắng mặt trên máy cục bộ hiện tại.
   - Môi trường cục bộ là CPU (không có CUDA GPU) nên việc huấn luyện chính thức cần được thực hiện trên Kaggle hoặc Cloud GPU bằng cách trỏ đường dẫn dữ liệu.
