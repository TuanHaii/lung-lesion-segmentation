# DATASET SPECIFICATION DOCUMENT (v1.0)

## 1. Metadata tài liệu

| Trường | Giá trị |
| :--- | :--- |
| **Mã task** | DATA-01 |
| **Tên tài liệu** | Dataset Specification Document |
| **Tên đề tài** | Phân đoạn tổn thương CT phổi bằng kiến trúc Attention U-Net |
| **Người phụ trách** | Đông |
| **Phiên bản** | v1.0 |
| **Trạng thái** | Baseline specification |
| **Phạm vi** | Chuẩn hóa dữ liệu CT phổi 3D, trích xuất lát cắt axial 2D và chuẩn bị dữ liệu cho huấn luyện mô hình |
| **Các task liên quan** | DATA-02, QA-02, QA-03, DATA-05 |

---

## 2. Tổng quan và tính hợp pháp nguồn dữ liệu

### 2.1. Định danh bộ dữ liệu
Bộ dữ liệu nghiên cứu chính là **COVID-19 CT Lung and Infection Segmentation Dataset**, được công bố trên Zenodo và được phân phối lại trên Kaggle. Phiên bản Zenodo chứa 20 CT scan COVID-19 có gán nhãn, bao gồm nhãn phổi trái, phổi phải và vùng nhiễm trùng/tổn thương. Nhãn được hai bác sĩ chẩn đoán hình ảnh thực hiện và được một bác sĩ có kinh nghiệm kiểm tra xác nhận.

| Thuộc tính | Đặc tả |
| :--- | :--- |
| **Tên dataset** | COVID-19 CT Lung and Infection Segmentation Dataset |
| **Nguồn chính** | Zenodo |
| **Nguồn tham chiếu** | Kaggle và các bản sao phục vụ nghiên cứu |
| **DOI tham chiếu** | 10.5281/zenodo.3757475 |
| **DOI bản ghi dữ liệu** | 10.5281/zenodo.3757476 |
| **Số lượng bệnh nhân/ca gán nhãn** | 20 CT volumes |
| **Dạng dữ liệu gốc** | 3D CT volumes và ground-truth masks |
| **Định dạng** | NIfTI: `.nii`, `.nii.gz` |
| **Mặt phẳng xử lý** | Axial (mặt phẳng cắt ngang) |
| **Đối tượng nhãn** | Phổi trái, phổi phải và nhiễm trùng/tổn thương |
| **Đơn vị cường độ** | Hounsfield Unit (HU), khi metadata DICOM/NIfTI cho phép xác định |
| **Mục đích sử dụng** | Nghiên cứu học thuật, phát triển và đánh giá mô hình phân đoạn |

*Ghi chú:* Số lượng 20 CT volumes được xem là số ca gán nhãn của nguồn Zenodo. Số lượng lát cắt 2D sau trích xuất phải được xác định bằng manifest thực tế, không suy ra trực tiếp từ số volume hoặc từ các bản sao Kaggle.

### 2.2. Giấy phép và điều kiện sử dụng
Nguồn Zenodo được ghi nhận dưới giấy phép **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Việc sử dụng dữ liệu tuân thủ các điều kiện:
* Sử dụng cho nghiên cứu học thuật và phát triển mô hình Attention U-Net.
* Không sử dụng dữ liệu để đưa ra chẩn đoán lâm sàng trực tiếp.
* Không tuyên bố mô hình đã được phê duyệt cho mục đích y tế hoặc triển khai lâm sàng.
* Không phát hành lại dữ liệu gốc nếu không đáp ứng đầy đủ điều kiện của giấy phép và chính sách của kho dữ liệu.
* Lưu giữ thông tin DOI, giấy phép, phiên bản dữ liệu và checksum trong hồ sơ provenance.

### 2.3. Yêu cầu provenance
Mỗi lần tiếp nhận dữ liệu phải ghi nhận tối thiểu:
1. URL hoặc DOI nguồn tải.
2. Ngày tải dữ liệu.
3. Phiên bản hoặc mã bản ghi của nguồn.
4. Tên file gốc.
5. Kích thước file.
6. Checksum SHA-256 / MD5.
7. Số lượng volume và mask.
8. Thông tin định hướng không gian NIfTI.
9. Các biến đổi đã áp dụng trong pipeline.

---

## 3. Tiêu chí chọn lọc và trích xuất lát cắt 2D

### 3.1. Mục tiêu trích xuất
Dữ liệu 3D được chuyển thành tập lát cắt 2D theo mặt phẳng ngang (axial plane) để huấn luyện và đánh giá mô hình phân đoạn tổn thương phổi:
* **Quy mô đầu ra mục tiêu:** 1.500–2.000 lát cắt 2D đạt chuẩn, mỗi lát cắt gồm một ảnh CT đã tiền xử lý và một ground-truth mask tương ứng.
* **Nguyên tắc phân định đơn vị:** Đơn vị độc lập để phân chia dữ liệu là bệnh nhân hoặc volume, không phải lát cắt. Lát cắt 2D chỉ là đơn vị huấn luyện sau khi phân chia bệnh nhân đã hoàn tất.

### 3.2. Tiêu chí đưa vào (Inclusion Criteria)
* Lát cắt axial hợp lệ của CT ngực, có biểu hiện rõ của nhu mô phổi hoặc vùng lồng ngực liên quan.
* Ảnh CT đọc được, không bị hỏng file hoặc sai kích thước.
* Có mask tương ứng, căn chỉnh không gian hoàn toàn với ảnh và cùng kích thước mặt phẳng $x$-$y$.
* Ưu tiên lát cắt chứa tổn thương: Ground-Glass Opacity (GGO), đông đặc (Consolidation).
* Bổ sung tỷ lệ lát cắt phổi lành đạt chuẩn để mô hình học nền giải phẫu và giảm thiên lệch.
* Thông tin không gian hợp lệ (spacing, orientation, slice index xác định được).

### 3.3. Tiêu chí loại bỏ (Exclusion Criteria)
* Nằm ngoài lồng ngực (cổ, bụng) hoặc chỉ chứa nền đen ngoài cơ thể.
* Nhiễu nghiêm trọng, artifact kim loại hoặc cử động làm mất cấu trúc tổn thương.
* Lỗi đọc NIfTI, lỗi giải nén hoặc lệch checksum.
* Ảnh và mask không cùng không gian tọa độ hoặc mask rỗng khi thuộc nhóm lesion-containing slice.
* Mask chứa giá trị bất thường ngoài tập nhãn quy định hoặc biên mask bị lỗi số hóa.
* Lát cắt bị trùng lặp hoặc không thể xác định patient ID/volume nguồn.

### 3.4. Phân bố lát cắt mục tiêu
Duy trì sự cân bằng tương đối giữa:
* Lát cắt có tổn thương và lát cắt phổi lành.
* Tổn thương kính mờ (GGO) và tổn thương đông đặc (Consolidation).
* Kích thước tổn thương: nhỏ ($< 5\%$ diện tích phổi), trung bình và lớn.

> **Ràng buộc:** Tuyệt đối không cân bằng bằng cách sao chép lát cắt giữa các bệnh nhân hoặc đưa cùng một lát cắt vào nhiều split. Oversampling chỉ được thực hiện bên trong tập Training sau khi đã chia patient-level.

### 3.5. Quy trình trích xuất axial
1. Đọc CT volume và mask bằng thư viện NIfTI (`nibabel`).
2. Kiểm tra shape, affine, orientation, voxel spacing và thứ tự trục.
3. Chuẩn hóa định hướng về quy ước axial thống nhất.
4. Lọc lát cắt theo các chỉ số: diện tích nhu mô phổi, diện tích tổn thương, tỷ lệ mask trên phổi.
5. Gán mã định danh bất biến cho bệnh nhân, volume và lát cắt.
6. Ghi thông tin vào manifest trước khi thực hiện resize hoặc augmentation.
7. Xuất ảnh và mask đã xử lý vào `data/processed/`.

### 3.6. Định danh mẫu và trường Manifest
Quy ước định danh mẫu:
$$\text{sample\_id} = \text{\{patient\_id\}\_\{volume\_id\}\_z\{slice\_index:04d\}}$$
*(Ví dụ: `P001_V001_z0128`)*

| Trường | Mô tả |
| :--- | :--- |
| `sample_id` | Định danh duy nhất của lát cắt |
| `patient_id` | Định danh bệnh nhân (đã ẩn danh) |
| `volume_id` | Định danh CT volume |
| `slice_index` | Chỉ số lát cắt axial trong volume |
| `image_path` | Đường dẫn ảnh đã xử lý |
| `mask_path` | Đường dẫn mask tương ứng |
| `original_shape` | Kích thước lát cắt gốc ($H \times W$) |
| `processed_shape` | Kích thước sau tiền xử lý ($256 \times 256$ hoặc $512 \times 512$) |
| `spacing_xy` | Pixel spacing trong mặt phẳng axial |
| `lesion_present` | 0 (phổi lành) hoặc 1 (có tổn thương) |
| `lesion_area_px` | Diện tích tổn thương tính theo pixel |
| `lung_area_px` | Diện tích vùng phổi |
| `source_doi` | DOI nguồn dữ liệu |
| `preprocessing_version` | Phiên bản tiền xử lý (`v1.0`) |
| `split` | Phân bổ tập: `train`, `validation` hoặc `test` |
| `qa_status` | Trạng thái kiểm duyệt QA (`passed`, `flagged`) |

---

## 4. Đặc tả kỹ thuật tiền xử lý

### 4.1. HU Windowing
Ảnh CT cắt dải cường độ theo cửa sổ nhu mô phổi $[-1000, 400]$ HU để tối ưu độ tương phản giữa mô lành và tổn thương đông đặc/GGO:
$$HU_{\text{clip}} = \min(\max(HU, -1000), 400)$$

### 4.2. Chuẩn hóa ảnh (Normalization)
Sau khi clip dải HU, ảnh được co dãn tuyến tính (Min-Max scaling) về đoạn $[0, 1]$:
$$I_{\text{norm}} = \frac{HU_{\text{clip}} - (-1000)}{400 - (-1000)} = \frac{HU_{\text{clip}} + 1000}{1400}$$
* Kiểu dữ liệu lưu trữ chuẩn: `float32`.
* Không áp dụng z-score cục bộ theo từng slice trong baseline nhằm giữ tính nhất quán tương đối của giá trị HU giữa các bệnh nhân.

### 4.3. Kích thước không gian & Nội suy (Resizing)
* **Kích thước chuẩn:** $256 \times 256$ pixels (kích thước thay thế: $512 \times 512$ pixels).
* **Thuật toán nội suy ảnh CT:** Bilinear hoặc Bicubic interpolation.
* **Thuật toán nội suy Mask:** **Nearest Neighbor** (bắt buộc để không làm nhòe biên nhãn và không sinh giá trị trung gian).

### 4.4. Nhị phân hóa Ground-Truth Mask (Binary Mask)
Ground-truth mask đầu ra là mask nhị phân nghiêm ngặt:
$$M_{\text{bin}}(x, y) = \begin{cases} 1, & \text{nếu pixel thuộc vùng tổn thương (Infection / GGO / Consolidation)} \\ 0, & \text{nền và nhu mô phổi không tổn thương} \end{cases}$$
* Nhãn phổi trái và phổi phải không được coi là tổn thương.
* Sau khi resize bằng Nearest Neighbor, toàn bộ giá trị mask phải thuộc tập $\{0, 1\}$ tuyệt đối.

### 4.5. Data Augmentation
Augmentation không tạo sẵn vào dữ liệu nền của DATA-01 mà chỉ áp dụng on-the-fly trong tập huấn luyện:
* Chỉ áp dụng cho tập **Training** sau khi đã phân chia patient-level; tập Validation và Test giữ nguyên bản.
* Phép biến đổi hình học (xoay nhẹ $\pm 15^\circ$, lật ngang, elastic transform) phải áp dụng đồng thời và đồng bộ cho cả ảnh lẫn mask.

---

## 5. Chiến lược phân chia Patient-level & Chống Data Leakage

### 5.1. Tỷ lệ phân chia

| Split | Tỷ lệ mục tiêu | Quy tắc áp dụng |
| :--- | :---: | :--- |
| **Training** | 70% | Huấn luyện mạng U-Net / Attention U-Net / SegNet |
| **Validation** | 15% | Đánh giá checkpoint, early stopping, tinh chỉnh siêu tham số |
| **Test** | 15% | Đánh giá độc lập cuối cùng, niêm phong đến tuần W5 |

*Số lượng bệnh nhân thực tế từ 20 volume gốc được chia theo số nguyên (ví dụ: 14 train / 3 val / 3 test) và cố định bằng random seed (seed = 2026).*

### 5.2. Nguyên tắc chống rò rỉ dữ liệu (Zero Data Leakage)
1. **Cô lập cấp bệnh nhân:** Một `patient_id` chỉ xuất hiện trong duy nhất một split.
2. Toàn bộ các lát cắt của cùng một volume/bệnh nhân phải nằm chung trong cùng một tập.
3. Tuyệt đối không xáo trộn ngẫu nhiên (random shuffle) ở cấp độ slice.
4. Tập Test bị khóa hoàn toàn, không tham gia vào chuẩn hóa hay tuning hyperparameter.

### 5.3. Cấu trúc lưu trữ Split & Bàn giao QA-03
* File phân bổ chính: `data/splits/splits.json` và `data/splits/splits.csv`.
* **Điều kiện nghiệm thu QA-03:** Hào (QA) thực hiện kiểm toán độc lập xác nhận giao thoa giữa 3 tập bằng rỗng ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$) trước khi mở khóa huấn luyện.

```json
{
  "dataset_name": "COVID-19 CT Lung and Infection Segmentation Dataset",
  "split_version": "v1.0",
  "seed": 2026,
  "unit": "patient",
  "ratios": { "train": 0.70, "validation": 0.15, "test": 0.15 },
  "train": ["P001", "P002", "..."],
  "validation": ["P015", "P016", "P017"],
  "test": ["P018", "P019", "P020"]
}
```

---

## 6. Sơ đồ tổ chức cấu trúc thư mục dữ liệu

```text
project_root/
├── data/
│   ├── raw/
│   │   ├── source_zenodo/
│   │   │   ├── ct_volumes/
│   │   │   │   └── <volume_id>.nii.gz
│   │   │   ├── ground_truth_masks/
│   │   │   │   └── <volume_id>_mask.nii.gz
│   │   │   ├── source_metadata.json
│   │   │   └── checksums.sha256
│   │   └── provenance/
│   │       ├── download_log.csv
│   │       └── license.txt
│   ├── processed/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── validation/
│   │   │   └── test/
│   │   ├── masks/
│   │   │   ├── train/
│   │   │   ├── validation/
│   │   │   └── test/
│   │   └── manifests/
│   │       ├── clean_slices_manifest.csv
│   │       └── processed_manifest.json
│   └── splits/
│       ├── splits.json
│       ├── summary_split.csv
│       └── leakage_check_report.txt
├── docs/
│   └── Dataset_Specification_Doc.md
└── reports/
    └── QA-03/
        └── QA_Report_Data_Leakage_Zero.md
```

---

## 7. Checklist nghiệm thu và chuyển giao

### 7.1. Bảng chuyển giao công việc

| Task kế tiếp | Người nhận | Nội dung chuyển giao từ DATA-01 |
| :--- | :--- | :--- |
| **DATA-02** | Đông | Cấu trúc thư mục `data/raw/`, link tải Zenodo và quy trình kiểm tra checksum MD5/SHA256. |
| **QA-02** | Hào | Danh sách volume thô, tiêu chuẩn affine/orientation và sanity check nhãn mask. |
| **DATA-05** | Đông | Cửa sổ HU $[-1000, 400]$, công thức Min-Max scaling $[0, 1]$. |
| **QA-03** | Hào | Cấu trúc `splits.json`, danh sách Patient ID và script kiểm tra zero data leakage. |

### 7.2. Điều kiện hoàn thành Task DATA-01
* [x] Tài liệu `Dataset_Specification_Doc.md` đã được biên soạn hoàn chỉnh và đẩy lên `docs/`.
* [x] Đã xác định rõ giấy phép mở CC BY 4.0 và 20 volumes CT từ Zenodo.
* [x] Đã thống nhất toàn bộ tham số tiền xử lý: HU $[-1000, 400]$, Min-Max $[0, 1]$, resize ảnh/mask (Nearest Neighbor cho mask).
* [x] Đã thiết lập nguyên tắc phân chia 70/15/15 theo Patient-level để bảo vệ tính toàn vẹn nghiên cứu.
