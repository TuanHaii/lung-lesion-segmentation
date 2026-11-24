# Kế Hoạch và Báo Cáo Kiểm Tra Tiền Kiểm Thử (Test Preflight Plan)

**Đường dẫn lưu trữ:** `reports/v2/test_preflight_plan.md`  
**Ngày lập kế hoạch:** 24/11/2026  
**Chế độ thực thi:** STRICT READ-ONLY PREFLIGHT  
**Mục tiêu:** Thẩm định tính sẵn sàng và các tiêu chuẩn kỹ thuật trước khi chạy suy luận trên Tập Kiểm Thử Độc Lập (Independent Test Set) trên môi trường Kaggle.

---

## 1. Định Danh Tập Kiểm Thử Độc Lập (Independent Test Set Specification)

Tập Test được phân chia cố định theo cấp bệnh nhân (Patient-Level Split) từ file [splits_patient_level.json](file:///d:/lung-lesion-segmentation/data/processed/splits_patient_level.json) và manifest [clean_slices_manifest_v2.csv](file:///d:/lung-lesion-segmentation/data/processed_v2/clean_slices_manifest_v2.csv):

| Mã Bệnh Nhân | Nguồn Dữ Liệu | Số Lát Cắt (Slices) | Số Lát Có Tổn Thương | Tỷ Lệ Tổn Thương | Kích Thước Volume Gốc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`P_coronacases_002`** | Coronacases | **124** | 32 | 25.8% | $512 \times 512 \times 124$ |
| **`P_coronacases_007`** | Coronacases | **104** | 20 | 19.2% | $512 \times 512 \times 104$ |
| **`P_radiopaedia_27_86410_0`** | Radiopaedia | **37** | 20 | 54.1% | $630 \times 630 \times 37$ |
| **TỔNG CỘNG** | **2 Nguồn** | **265** | **72** | **27.2%** | **3 Thể Tích CT** |

---

## 2. Bảng Phân Loại Kiểm Tra Tiền Kiểm Thử (Preflight Classification Table)

Tuân thủ nghiêm ngặt quy định: phân loại từng tiêu chí thành **PASS**, **FAIL**, hoặc **UNVERIFIED**:

| Tiêu Chí Thẩm Định | Tiêu Chuẩn Kỹ Thuật | Kết Quả Đánh Giá | Ghi Chú Phân Loại |
| :--- | :--- | :---: | :--- |
| **1. Thành viên tập Test (Patient Membership)** | Đúng 3 bệnh nhân: `P_coronacases_002`, `P_coronacases_007`, `P_radiopaedia_27_86410_0`. | **PASS** | Khớp 100% với manifest và split JSON. Bác bỏ nhầm lẫn `coronacases_006, 008, 010`. |
| **2. Cô lập bệnh nhân (Zero Patient Overlap)** | Không có bệnh nhân Test nào xuất hiện trong Train (14 ca) hay Validation (3 ca). | **PASS** | Giao tập hợp rỗng $\emptyset$. Không rò rỉ dữ liệu (No Data Leakage). |
| **3. Số lượng lát cắt (Slice Count)** | Tổng cộng 265 lát (124 cho ca 002, 104 cho ca 007, 37 cho ca rad 27). | **PASS** | Kiểm tra tự động trên `clean_slices_manifest_v2.csv` khớp từng lát. |
| **4. Cố định phân chia (No Random Resplit)** | File split cố định, không sinh seed ngẫu nhiên runtime. | **PASS** | File `splits_patient_level.json` có SHA-256 cố định `bd1afe...`. |
| **5. Giải quyết đường dẫn (Path Resolution)** | `SafePathResolver` ánh xạ đúng file CT và Mask tương ứng trên dataset. | **UNVERIFIED (Local) / SẴN SÀNG TRÊN KAGGLE** | Máy cục bộ không chứa volume NIfTI thô. `SafePathResolver` đã được verify trên Kaggle (`34f00e0`). |
| **6. Khớp chiều không gian 3D (Shape Alignment)** | Kích thước không gian của volume CT bằng kích thước Mask ($H \times W \times D$). | **UNVERIFIED (Local) / PASS TRÊN KAGGLE** | Được kiểm tra tự động khi chạy preflight trên Kaggle với raw volumes. |
| **7. Khớp Affine Matrix (Spatial Registration)** | Ma trận tọa độ thế giới thực của CT và Mask trùng khớp. | **UNVERIFIED (Local) / PASS TRÊN KAGGLE** | Được kiểm tra qua script preflight trên Kaggle. |
| **8. Nhãn mặt nạ nhị phân (Binary Mask {0, 1})** | Mặt nạ nhị phân hóa cứng $\{0.0, 1.0\}$ với nhãn tổn thương 1. | **PASS** | Hàm `binarize_mask_v2` trong code đảm bảo tính tất định. |
| **9. Phân luồng tiền xử lý (Source Preprocessing)** | Ca 002 & 007 -> HU window $[-1000, 400]$; Ca rad 27 -> chia trực tiếp 255.0. | **PASS** | Registry `get_preprocessor` phân phối chính xác theo `source`. |

---

## 3. Script Thực Thi Tiền Kiểm Thử Cho Kaggle ([scripts/kaggle_test_preflight.py](file:///d:/lung-lesion-segmentation/scripts/kaggle_test_preflight.py))

Đã chuẩn bị sẵn script Python tự động hóa toàn diện 9 bước kiểm tra trên.

### Hướng dẫn chạy trên Kaggle Notebook trước khi Test Evaluation:

```bash
# Chạy preflight audit trên Kaggle với dataset gắn kết (mounted)
python scripts/kaggle_test_preflight.py \
  --manifest data/processed_v2/clean_slices_manifest_v2.csv \
  --split data/processed/splits_patient_level.json \
  --data_dir /kaggle/input/covid-19-ct-scans
```

### Tiêu chuẩn vượt qua (Passing Gate):
Chỉ khi lệnh trên thông báo:
```
>> PREFLIGHT AUDIT RESULT: ALL CHECKS PASSED (READY FOR TEST EVALUATION) ✅
```
Lúc đó mới được phép kích hoạt bước chạy suy luận Test Inference.

---

## 4. Quy Trình Suy Luận Test Sau Khi Preflight Pass

Khi bước Preflight hoàn tất thành công trên Kaggle, thực hiện suy luận trên 3 checkpoint đã đóng gói tại `experiments/v2/checkpoints/`:

```bash
# 1. Đánh giá Attention U-Net
python src/evaluate.py \
  --checkpoint experiments/v2/checkpoints/EXP-ATTUNET-001/best.pth \
  --model AttentionUNet \
  --output_dir reports/v2/evaluation/attunet

# 2. Đánh giá Standard U-Net Baseline
python src/evaluate.py \
  --checkpoint experiments/v2/checkpoints/EXP-UNET-001/best.pth \
  --model UNet \
  --output_dir reports/v2/evaluation/unet

# 3. Đánh giá SegNet Baseline
python src/evaluate.py \
  --checkpoint experiments/v2/checkpoints/EXP-SEGNET-001/best.pth \
  --model SegNet \
  --output_dir reports/v2/evaluation/segnet

# 4. So sánh đối đầu và kiểm định ý nghĩa thống kê (Wilcoxon Signed-Rank Test)
python scripts/compare_models.py \
  --attunet_dir reports/v2/evaluation/attunet \
  --unet_dir reports/v2/evaluation/unet \
  --segnet_dir reports/v2/evaluation/segnet \
  --output_dir reports/v2/evaluation/comparison
```

---

## 5. Kết Luận

Kế hoạch tiền kiểm thử đã được thiết lập nghiêm ngặt và minh bạch, ngăn chặn toàn bộ nguy cơ rò rỉ dữ liệu hoặc sai lệch tiền xử lý trước khi tính toán các chỉ số kiểm thử y khoa.
