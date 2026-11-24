# Báo Cáo Đối Chiếu Tiền Xử Lý Dữ Liệu Nguồn Kép (Dual-Source Preprocessing Reconciliation)

**Thư mục lưu trữ:** `reports/v2_provenance_reconciliation/preprocessing_reconciliation.md`  
**Ngày thực hiện:** 24/11/2026  
**Chế độ thực thi:** STRICT READ-ONLY RECONCILIATION  
**Trạng thái đối chiếu:** **PASS** ✅  

---

## 1. Mục Tiêu Kiểm Tra

Kiểm tra trực tiếp mã nguồn tiền xử lý tại gói `src/data/v2/preprocessing/` để làm rõ:
1. Cơ chế tiền xử lý riêng biệt cho nguồn **Coronacases** (ảnh NIfTI gốc thang HU).
2. Cơ chế tiền xử lý riêng biệt cho nguồn **Radiopaedia** (ảnh web export thang uint8 [0, 255]).
3. Xác minh tính chuẩn hóa cường độ voxel, kích thước không gian và binarize nhãn tổn thương.
4. Đối chiếu với nhãn `preprocessing_version: "v2_dual_source"` được ghi trong cả 3 checkpoint.

---

## 2. Phân Tích Kỹ Thuật Hai Nguồn Dữ Liệu CT

Tập dữ liệu bài toán bao gồm hai nguồn ảnh CT có bản chất vật lý và dải động cường độ hoàn toàn khác nhau:

| Đặc Tính | Nguồn Coronacases (10 bệnh nhân) | Nguồn Radiopaedia (10 bệnh nhân) |
| :--- | :--- | :--- |
| **Định dạng file** | NIfTI 3D volume (`.nii.gz` / `.nii`) | NIfTI 3D volume (`.nii`) |
| **Kiểu dữ liệu voxel gốc** | `int16` hoặc `float32` mang giá trị HU chuẩn | `uint8` hoặc tiền chuẩn hóa thang xám `[0, 255]` |
| **Dải động cường độ gốc** | Trải rộng từ $-1024$ đến $+1000+$ Hounsfield Units | Trải rộng từ $0$ đến $255$ (đã bị windowing trên web) |
| **Nhu mô phổi & Tổn thương** | Không khí: $\approx -1000$ HU, Kính mờ (GGO): $\approx -600 \to -400$ HU | Toàn bộ mô đã được co cụm trong khoảng $[0, 255]$ |

---

## 3. Đối Chiếu Triển Khai Mã Nguồn Thực Tế (Code-Level Verification)

### 3.1. Tiền xử lý Coronacases (`src/data/v2/preprocessing/coronacases.py`)
Mã nguồn cài đặt lớp `CoronacasesPreprocessor`:
```python
class CoronacasesPreprocessor(BasePreprocessor):
    def __init__(self, min_hu: float = -1000.0, max_hu: float = 400.0):
        self.min_hu = float(min_hu)
        self.max_hu = float(max_hu)
        self.denom = self.max_hu - self.min_hu  # 1400.0

    def preprocess(self, image: np.ndarray, metadata=None) -> np.ndarray:
        img_float = image.astype(np.float32)
        clipped = np.clip(img_float, self.min_hu, self.max_hu)
        normalized = (clipped - self.min_hu) / self.denom
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)
```
- **Nguyên lý:** Áp dụng cửa sổ nhu mô phổi tiêu chuẩn (Pulmonary Windowing) $[-1000, 400]$ HU. Các giá trị dưới $-1000$ HU (không khí ngoài cơ thể) bị cắt cụt về 0.0; các giá trị trên $+400$ HU (xương sườn, cản quang) bị cắt cụt về 1.0. Dải kính mờ COVID-19 được phóng đại tuyến tính trong khoảng $[0.0, 1.0]$.

### 3.2. Tiền xử lý Radiopaedia (`src/data/v2/preprocessing/radiopaedia.py`)
Mã nguồn cài đặt lớp `RadiopaediaPreprocessor`:
```python
class RadiopaediaPreprocessor(BasePreprocessor):
    def __init__(self, max_value: float = 255.0):
        self.max_value = float(max_value)

    def preprocess(self, image: np.ndarray, metadata=None) -> np.ndarray:
        img_float = image.astype(np.float32)
        # Nếu đã ở dạng [0, 1] thì giữ nguyên an toàn
        if float(img_float.max()) <= 1.0 and float(img_float.min()) >= 0.0:
            return np.clip(img_float, 0.0, 1.0).astype(np.float32)
        normalized = img_float / self.max_value
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)
```
- **Tại sao Radiopaedia KHÔNG sử dụng HU Windowing?**  
  Tài liệu code giải thích rõ ràng:  
  *Nếu áp dụng sai công thức HU $[-1000, 400]$ lên ảnh Radiopaedia $[0, 255]$, giá trị voxel sẽ bị co ép vào khoảng hẹp $\left[\frac{0 - (-1000)}{1400}, \frac{255 - (-1000)}{1400}\right] = [0.714, 0.896]$, làm mất hoàn toàn độ tương phản và biến toàn bộ ảnh thành một màu xám đục.*  
  Do đó, `RadiopaediaPreprocessor` áp dụng chia trực tiếp cho 255.0 (`image / 255.0`), bảo toàn 100% độ tương phản mô mềm và viền tổn thương.

---

## 4. Cơ Chế Điều Phối và Tiền Xử Lý Không Gian Đồng Bộ

1. **Bộ điều phối động (`src/data/v2/preprocessing/registry.py`):**
   ```python
   PREPROCESSOR_REGISTRY = {
       "coronacases": CoronacasesPreprocessor(min_hu=-1000.0, max_hu=400.0),
       "radiopaedia": RadiopaediaPreprocessor(max_value=255.0),
   }
   ```
   Hàm `get_preprocessor(source)` tự động kích hoạt bộ tiền xử lý tương ứng dựa trên cột `source` trong manifest.

2. **Chuẩn hóa không gian và mặt nạ tổn thương (`src/data/v2/preprocessing/common.py`):**
   - **Ảnh CT:** Nội suy song tuyến (Bilinear Interpolation) về độ phân giải $[256, 256]$.
   - **Ground Truth Mask:** Nội suy láng giềng gần nhất (Nearest-Neighbor Interpolation) để bảo toàn giá trị nhãn rời rạc.
   - **Binarization:** Hàm `binarize_mask_v2(mask, lesion_label=1)` ép nhãn tổn thương COVID-19 (nhãn 1) về strictly $\{0.0, 1.0\}$.

3. **Tách biệt Data Augmentation giữa các tập (`src/data/v2/dataset.py`):**
   - **Train:** Áp dụng Albumentations (HorizontalFlip $p=0.5$, ShiftScaleRotate $p=0.5$, ElasticTransform $p=0.3$).
   - **Validation / Test:** Tuyệt đối không biến dạng hình học, chỉ thực hiện `Resize(256, 256)`.

---

## 5. Đối Chiếu Với Checkpoint V2

Trích xuất thông tin siêu dữ liệu từ 3 file checkpoint trong `D:\TTNT\v2\checkpoints\`:
- `preprocessing_version`: `"v2_dual_source"`
- `lesion_label`: `1`
- `image_size`: `[256, 256]`
- `in_channels`: `1`, `out_channels`: `1`

Toàn bộ các tham số này khớp hoàn toàn với pipeline tiền xử lý V2 đã được kiểm toán.

---

## 6. Kết Luận Đối Chiếu Tiền Xử Lý

| Tiêu Chí Kiểm Tra | Kết Quả Thực Tế | Đánh Giá |
| :--- | :--- | :--- |
| **Coronacases HU Windowing** | Cửa sổ $[-1000, 400]$ HU chuẩn hóa tuyến tính sang $[0, 1]$. | **PASS** |
| **Radiopaedia Direct Scaling** | Chia tuyến tính 255.0, không áp dụng sai HU windowing. | **PASS** |
| **Độ phân giải & Binarize** | Ảnh $256 \times 256$, Mask $\{0, 1\}$ nhãn tổn thương COVID-19. | **PASS** |
| **Khớp nhãn Checkpoint** | Checkpoint ghi nhận đúng `v2_dual_source`. | **PASS** |

**KẾT LUẬN CUỐI CÙNG: PASS ✅**
