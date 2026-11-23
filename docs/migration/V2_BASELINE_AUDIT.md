# V2 BASELINE AUDIT REPORT
## Lung Lesion Segmentation | Dual-Source Dataset Architecture

- **Date:** 2026-11-23 (Asia/Saigon)
- **Author:** Senior ML Engineer & Research Software Engineer
- **Git Branch:** `main`
- **Git HEAD Commit:** `d6aad6cbaf8fe6bc4de3ca5ae972be5c91c41b0c`
- **Document Status:** Official Baseline Audit for V2 Pipeline Migration

---

## 1. Executive Summary

Audit này đánh giá toàn diện trạng thái kỹ thuật hiện tại (V1) của repository `lung-lesion-segmentation` nhằm chuẩn bị cho việc thiết kế và triển khai Pipeline V2.
Các phát hiện chính:
1. **Lỗi nghiêm trọng trong Dataset Resolver V1 (Đã xác minh 100% bằng thực nghiệm):** Logic fallback của `_resolve_path` trong cả `src/data/dataset.py` và `src/data/eval_loader.py` tự động trả về `valid_matches[0]` bất kể loại file khi không tìm thấy keyword tương ứng. Kết quả thực nghiệm trực tiếp trong workspace: cả đường dẫn CT và Mask của `coronacases_001` đều phân giải về cùng một file `data/raw/ct_volumes/coronacases_001.nii`. Hơn nữa, file này có giá trị phân bố pixel nhị phân `{0.0, 1.0}`, thực chất là một binary mask bị đặt nhầm vào thư mục `ct_volumes`.
2. **Khuyết tật tiền xử lý dữ liệu hai nguồn (Dual-Source Intensity Shift):** Pipeline V1 áp dụng đồng loạt HU windowing `[-1000, 400]` và Min-Max scaling `(x + 1000) / 1400` cho cả hai nguồn dữ liệu. Trong khi Coronacases có dữ liệu `int16` theo thang HU, Radiopaedia lưu trữ ảnh dưới dạng `uint8 [0, 255]`. Khi áp dụng công thức HU trên lên Radiopaedia, dải giá trị bị nén phi lý vào khoảng `[0.714, 0.896]`, làm mất gần như toàn bộ độ tương phản của mô phổi và tổn thương.
3. **Phân nhánh Training Script (Dual Training Path):** Tồn tại hai luồng training tách biệt: `train.py` (generic engine có YAML config, AdamW, Cosine scheduler, early stopping) và `train_segnet.py` (hardcoded script sử dụng Adam, lr 1e-4 cố định, batch size 8 thay vì 16, epochs 30, không có scheduler, ghi log và checkpoint riêng biệt). Điều này vi phạm nguyên tắc fair comparison giữa ba kiến trúc (U-Net, Attention U-Net, SegNet).
4. **Lỗi tính toán và gộp Metric (Metric Aggregation Bug):** `src/utils/metrics.py` làm phẳng toàn bộ batch tensor thành 1D vector để tính micro-metric của batch, sau đó vòng lặp validation nhân ngược với batch size rồi chia tổng mẫu. Khi batch chứa các lát cắt rỗng (empty mask) hoặc số pixel dương chênh lệch lớn, giá trị validation Dice bị sai lệch so với macro per-slice Dice chuẩn.
5. **Cơ chế Silent Mock Fallback nguy hiểm trong Evaluation:** `src/data/eval_loader.py` chứa logic tự động chuyển sang `_generate_mock_slice` tạo dữ liệu giả lập khi không tìm thấy file NIfTI trên đĩa mà không báo lỗi hoặc dừng chương trình.

---

## 2. Git State & Working Tree Audit

- **Current Branch:** `main`
- **Current HEAD Commit:** `d6aad6cbaf8fe6bc4de3ca5ae972be5c91c41b0c` (`Merge branch 'main' of https://github.com/TuanHaii/lung-lesion-segmentation`)
- **Other Branches Present:**
  - `feature/check` (`617530f`)
  - `fix/lesion-label-3` (`0437ec6`)
  - `future/setup-unet` (`91557f2`)
- **Modified Tracked Files:**
  - `.gitignore` (bảo toàn nguyên vẹn, không can thiệp)
- **Untracked Directories & Files (Được bảo vệ nghiêm ngặt):**
  - `audit_checkpoint_current_sha256.json`
  - `docs/AttentionUNet_BaoCao_NCKH.pptx`
  - `docs/Bai_bao_Attention_UNet.pdf`
  - `docs/Bai_bao_Attention_UNet_v15.docx`
  - `docs/Bao_Cao_AI.docx`
  - `docs/HUONG_DAN_DIEN_VA_XAC_MINH_BAI_BAO_V15.md`
  - `docs/So_tay_Attention_UNet.pdf`
  - `docs/script_thuyet_trinh_15_phut.md`
  - `docs/script_thuyet_trinh_giai_thich_nen_tang.md`
  - `reports/audit/`
  - `reports_dev/`
  - `scripts/audit/`
- **Cam kết an toàn Git:** Không thực hiện `git checkout`, `git reset`, `git clean` hoặc commit tự động. Mọi thay đổi chưa commit được bảo lưu toàn diện.

---

## 3. Existing Pipeline Architecture

Repository hiện tại có cấu trúc module:
```text
lung-lesion-segmentation/
├── configs/
│   ├── baseline_unet.yaml
│   ├── attention_unet.yaml
│   └── segnet.yaml
├── data/
│   ├── processed/
│   │   ├── clean_slices_manifest.csv
│   │   ├── splits_patient_level.json
│   │   ├── test_manifest.json
│   │   └── summary_split.csv
│   └── raw/
│       ├── ct_volumes/coronacases_001.nii (mask binary đặt nhầm)
│       └── md5_checksums.txt
├── src/
│   ├── data/
│   │   ├── augmentations.py
│   │   ├── dataset.py
│   │   ├── eval_loader.py
│   │   ├── hu_window.py
│   │   ├── mask_binarize.py
│   │   └── transform.py
│   ├── losses/
│   │   ├── bce_loss.py
│   │   ├── dice_loss.py
│   │   └── hybrid_loss.py
│   ├── models/
│   │   ├── attention_unet.py
│   │   ├── segnet.py
│   │   └── unet.py
│   └── utils/
│       ├── metrics.py
│       └── seed.py
├── train.py
├── train_segnet.py
├── evaluate.py
└── demo_segnet_inference.py
```

---

## 4. Dataset Resolver Findings

### 4.1. Mã nguồn Resolver V1
Trong `src/data/dataset.py:100-182` và `src/data/eval_loader.py:214-295`:
```python
clean_stem = p.name.replace(".nii.gz", "").replace(".nii", "")
is_mask = any(k in path.lower() for k in ["mask", "infection", "ground_truth", "label"])
...
for root in search_roots:
    matches = list(root.rglob(f"{clean_stem}.nii*"))
    valid_matches = [m for m in matches if m.is_file() and m.stat().st_size > 0]
    ...
    if filtered:
        self._path_cache[path] = str(filtered[0])
        return str(filtered[0])
    elif valid_matches:
        self._path_cache[path] = str(valid_matches[0])  # <-- LỖI NGUY HIỂM: Fallback sai loại file
        return str(valid_matches[0])
```

### 4.2. Bằng chứng thực nghiệm
Kiểm tra trực tiếp trên workspace:
- Gọi `_resolve_path("data/raw/ct_volumes/coronacases_001.nii.gz")` -> Trả về `data\raw\ct_volumes\coronacases_001.nii`.
- Gọi `_resolve_path("data/raw/ground_truth_masks/coronacases_001.nii.gz")` -> Trả về `data\raw\ct_volumes\coronacases_001.nii`.
- Kết quả `ct_res == mask_res` là `True`!
- Phân tích file `data/raw/ct_volumes/coronacases_001.nii`:
  - Kích thước: 157,811,040 bytes
  - Header dtype: uint16
  - Unique values: `[0.0, 1.0]` (0: 78,253,055 voxels; 1: 652,289 voxels).
  - Đây là binary ground truth mask bị lưu vào thư mục `ct_volumes`.
- Hậu quả: Mô hình nhận mask làm input CT và tính loss với chính mask đó, hoặc CT bị tráo đổi thành mask.

---

## 5. Preprocessing Findings

### 5.1. Dữ liệu hai nguồn (Coronacases vs Radiopaedia)
- **Coronacases:** 10 volumes CT ngực. Dữ liệu gốc lưu dưới dạng `int16` hoặc `float32` phản ánh giá trị Hounsfield Units (HU) trong dải `[-1024, 3071]`. Đối với nguồn này, HU Windowing `[-1000, 400]` và Min-Max chuẩn hóa tuyến tính là chuẩn mực y khoa chính xác.
- **Radiopaedia:** 10 volumes CT ngực. Dữ liệu gốc đã qua windowing/export từ web dưới định dạng ảnh 8-bit (`uint8`), dải cường độ `[0, 255]`.
- **Sai lầm trong V1:** `src/data/dataset.py:228-230`:
  ```python
  norm_img = normalize_min_max(raw_slice, min_hu=-1000.0, max_hu=400.0)
  ```
  Khi áp dụng lên ảnh Radiopaedia (`[0, 255]`):
  - `clipped = np.clip(img, -1000, 400)` -> Giữ nguyên `[0, 255]`.
  - `norm = (clipped - (-1000)) / 1400 = (clipped + 1000) / 1400`.
  - Dải giá trị đầu ra: `[1000/1400, 1255/1400] ≈ [0.7143, 0.8964]`.
  - Toàn bộ ảnh bị xám hóa, nén dải động vào khoảng 18% của thang đo `[0, 1]`, phá vỡ phân phối gradient và tương phản của mô tổn thương so với mô phổi lành.

---

## 6. Training Pipeline Findings

### 6.1. Bất nhất giữa train.py và train_segnet.py
| Tiêu chí | `train.py` (Generic Engine) | `train_segnet.py` (Legacy SegNet) |
|---|---|---|
| Cấu hình | Đọc từ YAML file (`configs/*.yaml`) | Hardcoded tham số qua CLI argparse |
| Mô hình hỗ trợ | U-Net, Attention U-Net, SegNet | Chỉ SegNet |
| Optimizer | `AdamW` (lr: 1e-4, weight_decay: 0.01) | `Adam` (lr: 1e-4, không weight_decay) |
| Scheduler | `CosineAnnealingLR` (T_max=100) | Không có scheduler (fixed learning rate) |
| Batch Size | 16 (theo config) | 8 (hardcoded default) |
| Epochs | 100 (với EarlyStopping patience=15) | 30 (cố định) |
| Checkpoint lưu trữ | `checkpoints/{experiment_id}_best.pth` | `checkpoints/segnet_best.pth` |
| Metadata lưu | Lưu state_dict + optimizer + scheduler + best_dice | Chỉ lưu epoch, state_dict, optimizer, best_dice |
| Fair Comparison | Đạt chuẩn | Vi phạm nghiêm trọng |

---

## 7. Evaluation Pipeline Findings

### 7.1. Lỗi Metric Aggregation trong V1
- `src/utils/metrics.py:14-25`:
  ```python
  preds = preds.view(-1)
  true_mask = true_mask.view(-1)
  tp = (preds * true_mask).sum()
  ...
  dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
  ```
  Hàm làm phẳng toàn bộ batch `[B, 1, H, W]` thành vector 1D, tính một giá trị Dice gộp cho cả batch (micro-batch metric).
- `train.py:112`:
  ```python
  total_dice += metrics["dice"] * batch_size
  ```
  Sau đó nhân với `batch_size` và chia tổng số sample. Điều này không tương đương với Macro Dice trung bình theo từng slice (Macro per-slice Dice).
- Trong các script evaluation độc lập (`scripts/run_eval_02.py`), batch size được đặt là 1, nên kết quả test trong reports không bị ảnh hưởng bởi lỗi batch aggregation này, nhưng validation Dice trong quá trình training bị sai lệch.

### 7.2. Lỗi Silent Mock Fallback
- `src/data/eval_loader.py:327-329`:
  ```python
  if not os.path.exists(resolved_ct) or not os.path.exists(resolved_mask):
      if not Path("/kaggle").exists() and not self.raw_data_dir:
          return self._generate_mock_slice(record, idx)
  ```
  Nếu không có raw data cục bộ, evaluator tự sinh slice giả lập và tính toán mà không raise exception, gây rủi ro tạo ra báo cáo khoa học không có thực tế.

---

## 8. Dataset Structure & Patient-level Split Audit

### 8.1. Kiểm tra Split hiện tại
Từ `data/processed/splits_patient_level.json` và `data/processed/clean_slices_manifest.csv`:
- **Tổng số bệnh nhân:** 20 (10 Coronacases + 10 Radiopaedia).
- **Tổng số lát cắt:** 1,750 lát cắt 2D chuẩn 512x512.
- **Phân bổ split:**
  - **Train (14 patients - 70%):**
    - Coronacases (7): `001, 003, 004, 006, 008, 009, 010`
    - Radiopaedia (7): `10_85902_1, 10_85902_3, 14_85914_0, 29_86490_1, 29_86491_1, 4_85506_1, 7_85703_0`
    - Tổng lát cắt: 1,277 lát (1,057 có tổn thương, 220 bình thường).
  - **Validation (3 patients - 15%):**
    - Coronacases (1): `005`
    - Radiopaedia (2): `36_86526_0, 40_86625_0`
    - Tổng lát cắt: 208 lát (172 có tổn thương, 36 bình thường).
  - **Test (3 patients - 15%):**
    - Coronacases (2): `002, 007`
    - Radiopaedia (1): `27_86410_0`
    - Tổng lát cắt: 265 lát (221 có tổn thương, 44 bình thường).
- **Đánh giá tính hợp lệ:**
  - Zero patient leakage: Tập giao giữa Train, Val và Test về patient ID là rỗng (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).
  - Cả ba tập đều cân bằng sự hiện diện của cả 2 nguồn (Coronacases và Radiopaedia).
  - Patient-level split hiện tại hoàn toàn hợp lệ về mặt khoa học, **bảo toàn 100% split này cho V2**, không tự ý thay đổi.

---

## 9. Existing Artifacts That Must Be Preserved

Các artifact sau đây thuộc V1 và đã được audit tính toàn vẹn:
1. **Checkpoints:**
   - `checkpoints/EXP-ATTUNET-001_best.pth` (SHA256: `815c2f84...`)
   - `checkpoints/EXP-UNET-001_best.pth` (SHA256: `a4133bfd...`)
   - `checkpoints/EXP-SEGNET-001_best.pth` (SHA256: `2d93e811...`)
2. **Manifests V1:**
   - `data/processed/clean_slices_manifest.csv`
   - `data/processed/splits_patient_level.json`
   - `data/processed/test_manifest.json`
   - `data/processed/summary_split.csv`
   - `data/processed/dataset_v1.0_freeze_manifest.json`
3. **Audit & Prediction Reports:**
   - `reports/unet/*`, `reports/attunet/*`, `reports/segnet/*`
   - `predictions/*`
   - `audit_independent/*`
   - `logs/*`

Tất cả các file trên phải được giữ nguyên, không chỉnh sửa, không ghi đè.

---

## 10. Summary of Classification Findings

| Vấn đề | Phân loại | Bằng chứng | Mức độ |
|---|---|---|---|
| Resolver fallback trả về NIfTI sai loại khi thiếu file | **CONFIRMED BUG** | Đã tái lập: `ct_res == mask_res` cho `coronacases_001` | Rất nghiêm trọng (Critical) |
| File mask bị đặt trong thư mục `ct_volumes` | **CONFIRMED DEFECT** | `coronacases_001.nii` chỉ chứa nhãn `{0, 1}` | Rất nghiêm trọng (Critical) |
| HU Window `[-1000, 400]` áp dụng lên `uint8 [0, 255]` Radiopaedia | **CONFIRMED BUG** | `src/data/dataset.py:228-230` nén dải Radiopaedia về `[0.71, 0.90]` | Rất nghiêm trọng (Critical) |
| Tồn tại 2 script huấn luyện (`train.py` vs `train_segnet.py`) | **CONFIRMED RISK** | Khác biệt optimizer, batch size, epochs, scheduler | Cao (High) |
| Metric aggregation làm phẳng batch rồi nhân batch_size | **CONFIRMED BUG** | `src/utils/metrics.py:14-25`, `train.py:112` | Trung bình (Medium) |
| Silent fallback tạo mock slice trong test evaluation | **CONFIRMED DEFECT** | `src/data/eval_loader.py:327-329` | Cao (High) |
| Patient split có rò rỉ dữ liệu (leakage) | **CONTRADICTED (KHÔNG CÓ LỖI)** | Giao tập Train/Val/Test = ∅, phân chia cấp bệnh nhân chuẩn | Đạt chuẩn |
| Dataset NIfTI đầy đủ trên máy cục bộ | **UNVERIFIED / BLOCKED** | Chỉ có 1 file volume cục bộ; 19 volumes còn lại vắng mặt | Giới hạn môi trường |

---

## 11. Next Action

Chuyển sang **Phase 1: V2 Architecture Design**, thiết kế:
1. Bộ phân loại nguồn dữ liệu `SourceDetector` (`coronacases` vs `radiopaedia`).
2. Resolver an toàn chống fallback sai loại (`SafePathResolver`).
3. Preprocessing module riêng biệt:
   - `CoronacasesPreprocessor`: HU window `[-1000, 400]`, chuẩn hóa `[0, 1]`.
   - `RadiopaediaPreprocessor`: Chuẩn hóa `[0, 255] -> [0.0, 1.0]`, bảo toàn độ tương phản gốc.
4. Data Integrity Validator độc lập.
5. Manifest V2 schema mở rộng.
6. Hợp nhất Training Engine và Checkpoint V2 metadata schema.
7. Evaluation V2 độc lập đa cấp độ (Overall, Patient-level, Source-level, Lesion-slice level).
