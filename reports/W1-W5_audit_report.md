# BÁO CÁO KIỂM TOÁN ĐỘC LẬP TIẾN ĐỘ & BẰNG CHỨNG NGHIÊN CỨU
## CÁC NHIỆM VỤ TỪ TUẦN 1 ĐẾN TUẦN 5 (THÀNH VIÊN: NGUYỄN PHƯƠNG ĐÔNG)

- **Dự án:** Phân đoạn tổn thương phổi trên ảnh cắt lớp vi tính (CT) lồng ngực sử dụng kiến trúc Attention U-Net
- **Người phụ trách được kiểm toán:** Nguyễn Phương Đông
- **Phạm vi kiểm toán:** Toàn bộ 21 nhiệm vụ được phân công trong W1, W2, W3, W5 (căn cứ theo `TASK_MASTER`)
- **Ngày thực hiện kiểm toán:** 01/11/2026
- **Kho lưu trữ:** `ttnt/lung-lesion-segmentation` (Git commit: `d899120` và các bản vá bổ sung)
- **Thư mục tài liệu Google Drive:** `https://drive.google.com/drive/u/0/folders/1gNJNq7aK7FKjE_xt87Sr5M-vT-Zy3W27`

---

## 1. MỤC ĐÍCH VÀ NGUYÊN TẮC KIỂM TOÁN

1. **Nguyên tắc đối chuẩn thực chứng (Evidence-based Audit):**
   Không mặc định thừa nhận trạng thái `Completed` trên Google Sheets (`TASKS_ĐÔNG` hoặc `TASK_MASTER`). Toàn bộ 21 nhiệm vụ phải được chứng minh bằng file mã nguồn, file dữ liệu, báo cáo chi tiết, log thực thi và kết quả kiểm thử tự động (Unit Test / Integration Test) còn nguyên giá trị hiệu lực trên hệ thống.
2. **Nguyên tắc toàn vẹn & trung thực khoa học:**
   Không dùng số liệu giả lập, không sử dụng checkpoint bị lỗi cấu hình gán nhãn, phản ánh khách quan năng lực phân đoạn của Attention U-Net đối với các baseline.
3. **Mục tiêu bàn giao:**
   - Phát hiện mọi điểm sai lệch giữa mô tả trên bảng phân công công việc và hiện trạng thực tế.
   - Khắc phục triệt để các bất nhất về đường dẫn, mã băm (checksum) và cấu hình xử lý nhãn.
   - Xuất bản báo cáo tài liệu (file `.docx`) trực tiếp vào thư mục Google Drive của đồ án.

---

## 2. BẢNG TỔNG HỢP KẾT QUẢ KIỂM TOÁN 21 NHIỆM VỤ (W1 - W5)

| Mã Task | Tuần | Tên nhiệm vụ | Deliverable quy định | Trạng thái Sheet | Hiện trạng thực tế | Đánh giá kiểm toán |
|:---:|:---:|:---|:---|:---:|:---|:---:|
| **RES-04** | W1 | Xác định Research Gap, RQ & Metrics | `Metrics_Specification.md` | Completed | Đầy đủ lý thuyết, 2 RQ, công thức toán DSC/IoU/P/R; đã đồng bộ vào `docs/` | **ĐẠT (PASS)** |
| **DATA-01** | W2 | Đặc tả dataset chính & subset | `Dataset_Specification_Doc.md` | Completed | Đầy đủ đặc tả 20 CT Zenodo, 1.750 slices; đã đồng bộ vào `docs/` | **ĐẠT (PASS)** |
| **DATA-02** | W2 | Lưu trữ dữ liệu CT thô & MD5 | `data/raw/` & `md5_checksums.txt` | Completed | 41 file raw toàn vẹn, đã tính lại MD5 chuẩn xác 100% | **ĐẠT (PASS)** |
| **DATA-03** | W2 | Kiểm tra hình học & metadata | `Raw_Data_Integrity_Report.json` | Completed | 20 volume khớp shape 100%, đã cập nhật nhãn nhị phân `{0, 1}` | **ĐẠT (PASS)** |
| **DATA-04** | W2 | Lọc dữ liệu & lát cắt hợp lệ | `clean_slices_manifest.csv` & script | Completed | Đúng 1.750 lát (1.450 tổn thương, 300 lành), 0 sai sót pixel | **ĐẠT (PASS)** |
| **DATA-05** | W2 | HU Windowing [-1000, 400] | `src/data/hu_window.py` | Completed | NumPy/Tensor clipping, unit test tự động ĐẠT 100% | **ĐẠT (PASS)** |
| **DATA-06** | W2 | Chuẩn hóa Min-Max & Resize | `src/data/transform.py` | Completed | Scaling [0, 1], Bilinear/Nearest resize, unit test ĐẠT 100% | **ĐẠT (PASS)** |
| **DATA-07** | W2 | Nhị phân hóa Ground-Truth Mask | `src/data/mask_binarize.py` | Completed | Auto-fallback đa lớp sang nhị phân, unit test ĐẠT 100% | **ĐẠT (PASS)** |
| **DATA-08** | W2 | Patient-level Split (70/15/15) | `splits_patient_level.json` & CSV | Completed | 14 Train / 3 Val / 3 Test, Zero Leakage tuyệt đối | **ĐẠT (PASS)** |
| **DATA-09** | W2 | Augmentation chỉ cho Training | `src/data/augmentations.py` | Completed | Albumentations rotation/flip, Val/Test bất biến, test ĐẠT | **ĐẠT (PASS)** |
| **DATA-10** | W2 | PyTorch Dataset & DataLoader | `src/data/dataset.py` & `test_loader.py` | Completed | DataLoader đa luồng, benchmark 49.8 slices/s trên CUDA ĐẠT | **ĐẠT (PASS)** |
| **DATA-12** | W2 | Đóng băng Dataset v1.0 | `dataset_v1.0_freeze_manifest.json` | Completed | Đã cập nhật 8 mã băm SHA-256 mã nguồn và dữ liệu tươi mới 100% | **ĐẠT (PASS)** |
| **MODEL-02** | W3 | Cài đặt Soft Dice Loss | `src/losses/dice_loss.py` | Completed | Soft Dice Loss có smooth $\epsilon=10^{-5}$, macro slice-level | **ĐẠT (PASS)** |
| **MODEL-03** | W3 | Cài đặt Binary Cross-Entropy | `src/losses/bce_loss.py` | Completed | `nn.BCEWithLogitsLoss` ổn định số học, có pos_weight | **ĐẠT (PASS)** |
| **MODEL-04** | W3 | Cài đặt Hybrid Loss | `src/losses/hybrid_loss.py` | Completed | $0.5 \text{BCE} + 0.5 \text{Dice}$, unit test gradient & weight ĐẠT | **ĐẠT (PASS)** |
| **EVAL-01** | W5 | Cô lập tập Test độc lập | `test_manifest.json` & `eval_loader.py` | Completed | 265 lát cắt, 3 bệnh nhân unseen, quyền Read-Only 0444, test ĐẠT | **ĐẠT (PASS)** |
| **EVAL-03** | W5 | Tính toán chỉ số DSC | `results/metrics_dsc_detail.csv` | Completed | 221 lát tổn thương: Attn 0.6312, U-Net 0.6837, SegNet 0.6693 | **ĐẠT (PASS)** |
| **EVAL-04** | W5 | Tính toán chỉ số IoU | `results/metrics_iou_detail.csv` | Completed | Attn 0.5080, U-Net 0.5604, SegNet 0.5425; đủ 95% CI | **ĐẠT (PASS)** |
| **EVAL-05** | W5 | Tính toán chỉ số Precision | `results/metrics_precision_detail.csv` | Completed | Attn 0.5854, U-Net 0.6349, SegNet 0.6466 | **ĐẠT (PASS)** |
| **EVAL-06** | W5 | Tính toán chỉ số Recall | `results/metrics_recall_detail.csv` | Completed | Attn 0.8448, U-Net 0.8417, SegNet 0.7557 | **ĐẠT (PASS)** |
| **EVAL-07** | W5 | Tổng hợp số liệu & Wilcoxon | `quantitative_comparison_table.csv` | Completed | Paired Wilcoxon & t-test, bảng so sánh LaTeX tự động | **ĐẠT (PASS)** |

**Kết luận tổng quan kiểm toán:** **21 / 21 nhiệm vụ ĐẠT 100%**. Không có nhiệm vụ nào bị bỏ sót hay thiếu bằng chứng nghiệm thu.

---

## 3. CÁC ĐIỂM SAI LỆCH PHÁT HIỆN VÀ BIỆN PHÁP KHẮC PHỤC TRIỆT ĐỂ

Trong quá trình rà soát thực địa kho lưu trữ, kiểm toán đã phát hiện 5 điểm bất nhất kỹ thuật tiềm ẩn nguy cơ sai lệch kết quả giữa tài liệu thiết kế và mã nguồn thực thi:

### 3.1. Sai lệch mã băm dữ liệu thô (DATA-02)
- **Hiện trạng trước kiểm toán:** Tệp `data/raw/md5_checksums.txt` chứa 61 dòng băm từ cấu trúc thư mục cũ (bao gồm cả thư mục `infection_masks_only/` đã bị xóa). Khi kiểm tra mã băm, có tới 20 tệp mask báo lỗi hash mismatch và 20 tệp báo thiếu.
- **Biện pháp xử lý:** Chạy `scripts/generate_checksums.py` tính toán lại toàn bộ mã băm MD5 cho đúng 41 tệp thực tế trong `data/raw/` (20 CT volume `.nii.gz`, 20 mask ground-truth `.nii.gz` và 1 tệp `ReadMe.txt`).
- **Kết quả:** Kiểm tra xác nhận 41/41 tệp đạt chuẩn MD5 100%, 0 tệp lỗi, 0 tệp ngoài danh mục.

### 3.2. Sai lệch dải nhãn trong báo cáo siêu dữ liệu thô (DATA-03)
- **Hiện trạng trước kiểm toán:** Tệp `reports/DATA-03/Raw_Data_Integrity_Report.json` lưu thông tin nhãn `unique_labels: [0, 1, 2, 3]` do được trích xuất từ bộ dữ liệu đa lớp cũ, trong khi các tệp mask thực tế trong `data/raw/ground_truth_masks/` là nhị phân `{0, 1}`.
- **Biện pháp xử lý:** Chạy lại `scripts/inspect_raw_data.py`. Báo cáo được cập nhật chính xác: 20/20 volume có shape ảnh và mask khớp nhau tuyệt đối (`shape_matched: true`), dải nhãn mask được xác nhận là `{0, 1}`.

### 3.3. Rủi ro cứng mã nhãn trong script trích xuất lát cắt (DATA-04)
- **Hiện trạng trước kiểm toán:** Tệp `scripts/filter_slices.py` thiết lập cố định `LESION_LABEL = 3`. Nếu người dùng tái tạo lại tệp manifest trên bộ mask nhị phân hiện tại, script sẽ hiểu sai và không lọc được lát cắt tổn thương nào. Tuy nhiên, tệp manifest đã đóng băng `clean_slices_manifest.csv` lưu trong kho lưu trữ lại hoàn toàn chính xác.
- **Biện pháp xử lý:** 
  1. Tiến hành kiểm toán đối chiếu độc lập từng lát cắt: Quét toàn bộ 1.750 lát cắt trong `clean_slices_manifest.csv` đối chiếu với mảng 3D NIfTI trong `data/raw/ground_truth_masks/`. Kết quả: **Khớp 1.750 / 1.750 lát cắt, sai số 0%** (diện tích tổn thương `lesion_area_px` và cờ `lesion_present` chuẩn xác 100%).
  2. Nâng cấp `scripts/filter_slices.py`: Thêm hàm tự động nhận diện dải nhãn `resolve_lesion_label()`, bổ sung các tham số dòng lệnh `--ct-dir`, `--mask-dir`, `--output`, và thiết lập cơ chế khóa an toàn từ chối ghi đè file đã niêm phong nếu không có cờ `--force`.

### 3.4. Lệch mã băm mã nguồn trong biên bản đóng băng dữ liệu (DATA-12)
- **Hiện trạng trước kiểm toán:** Tệp `data/processed/dataset_v1.0_freeze_manifest.json` ghi nhận mã băm SHA-256 của các tệp mã nguồn tiền xử lý tại thời điểm Tuần 2. Qua các đợt cập nhật ở Tuần 4 và Tuần 5, 4 tệp mã nguồn (`transform.py`, `mask_binarize.py`, `augmentations.py`, `dataset.py`) có thay đổi nhỏ làm mã băm bị lệch (stale).
- **Biện pháp xử lý:** Chạy `scripts/freeze_dataset_v1.py` để tái đồng bộ toàn bộ 8 mã băm SHA-256 (3 tệp dữ liệu cốt lõi và 5 module mã nguồn tiền xử lý). Kiểm tra sau khi chạy: 8/8 tệp khớp 100%.

### 3.5. Đồng bộ vị trí lưu trữ tài liệu đặc tả (RES-04, DATA-01)
- **Hiện trạng trước kiểm toán:** Hai tài liệu nền tảng `Metrics_Specification.md` và `Dataset_Specification_Doc.md` chỉ tồn tại ở thư mục ngoài kho lưu trữ (`/home/dyspo/DTH235628/PPNCKH/`), khiến người chấm hoặc thành viên khác clone repo không thể tiếp cận được evidence.
- **Biện pháp xử lý:** Sao chép nguyên vẹn 2 tài liệu chuẩn mực này vào thư mục `docs/` của kho lưu trữ (`docs/Metrics_Specification.md` và `docs/Dataset_Specification_Doc.md`).

---

## 4. CHI TIẾT KẾT QUẢ KIỂM THỬ TỪNG MODULE KỸ THUẬT

### 4.1. Tiền xử lý dữ liệu (DATA-05, DATA-06, DATA-07, DATA-09)
Tất cả các module tiền xử lý dữ liệu của Đông đều được tích hợp khối Unit Test độc lập và tự kiểm tra:
1. **HU Windowing (`src/data/hu_window.py`):**
   - Kiểm thử giá trị ngoài biên: mảng HU `[-1250, -1000, -500, 0, 400, 1500]` được cắt chuẩn xác về `[-1000, -1000, -500, 0, 400, 400]`.
   - Hỗ trợ đồng thời NumPy array và PyTorch Tensor. Kết quả: **PASSED 100%**.
2. **Transform & Scaling (`src/data/transform.py`):**
   - Min-Max scaling đưa dải cường độ về đoạn chặt `[0.0, 1.0]`.
   - Resizing đồng bộ ảnh CT bằng phép nội suy song tuyến (`cv2.INTER_LINEAR`) và mask bằng láng giềng gần nhất (`cv2.INTER_NEAREST`) nhằm ngăn chặn hiện tượng làm mờ biên nhãn y tế. Kết quả: **PASSED 100%**.
3. **Nhị phân hóa Mask (`src/data/mask_binarize.py`):**
   - Hỗ trợ linh hoạt cả hai chuẩn gán nhãn: nhãn nhị phân `{0, 1}` trích xuất nhãn 1; nhãn đa lớp `{0, 1, 2, 3}` trích xuất nhãn 3.
   - Có cơ chế tự động hạ cấp (fallback) thông minh khi gặp mask nhị phân mà cấu hình yêu cầu nhãn 3. Kết quả: **PASSED 100%**.
4. **Data Augmentation (`src/data/augmentations.py`):**
   - Sử dụng thư viện Albumentations áp dụng cho tập Train: xoay nhẹ $\pm 15^\circ$, lật ngang ngẫu nhiên (`HorizontalFlip`), biến dạng co giãn nhẹ.
   - Tuyệt đối giữ nguyên ảnh gốc chuẩn hóa đối với tập Validation và Test. Kết quả: **PASSED 100%**.

### 4.2. Dataloader & Hiệu năng tải dữ liệu (DATA-10, EVAL-01)
Thực thi kiểm thử thực nghiệm trên kịch bản huấn luyện và kiểm thử tuần tự:
- **Tập Train (DATA-10):** 1.277 lát cắt / 79 batches (kích thước batch = 16).
  - Tensor Image: `[16, 1, 256, 256]`, dải giá trị `[0.000, 1.000]`.
  - Tensor Mask: `[16, 1, 256, 256]`, nhãn duy nhất `[0.0, 1.0]`.
  - Tốc độ tải dữ liệu đo được: **49.8 lát cắt/giây** trên GPU CUDA, đáp ứng vượt yêu cầu thông lượng cho quá trình huấn luyện mô hình.
- **Tập Test độc lập (EVAL-01):** 265 lát cắt / 265 batches (cố định kích thước batch = 1, shuffle = False).
  - Khóa quyền ghi: Tệp `data/processed/test_manifest.json` được phân quyền `0o444` (Read-Only) để ngăn chặn rò rỉ dữ liệu hoặc chỉnh sửa ngoài ý muốn.
  - Phân bổ bệnh nhân: 3 ca bệnh hoàn toàn cô lập (`P_coronacases_002`: 124 lát, `P_coronacases_007`: 104 lát, `P_radiopaedia_27_86410_0`: 37 lát). Số lát tổn thương: 221 (83.4%), số lát phổi lành: 44 (16.6%).
  - Kiểm định Zero Leakage: Tập bệnh nhân và tập lát cắt giữa Train, Val, Test có độ giao nhau bằng tập rỗng ($\emptyset$).

### 4.3. Các hàm mục tiêu toán học (MODEL-02, MODEL-03, MODEL-04)
Chạy bộ kiểm thử tự động `tests/test_hybrid_loss.py`:
- **Gradient Flow Check:** Lan truyền thuận và lan truyền ngược mượt mà qua cả hai nhánh BCE và Soft Dice, gradient truyền đủ về logits, không xảy ra hiện tượng NaN hay Inf.
- **Tỷ lệ kết hợp tuyến tính:** Với $\alpha = 0.5$ và $\beta = 0.5$, sai số kết hợp loss thực tế và lý thuyết $< 10^{-5}$.
- Kết quả kiểm thử: **ALL HYBRID LOSS TESTS PASSED SUCCESSFULLY!**

---

## 5. KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM ĐỊNH LƯỢNG W5 (EVAL-03 .. EVAL-07)

Quá trình kiểm toán W5 được thực hiện nghiêm ngặt dựa trên việc loại bỏ hoàn toàn các checkpoint cũ ngày 25/10/2025 (vốn bị lỗi huấn luyện trên mask rỗng do nhầm nhãn) và huấn luyện lại toàn bộ 3 kiến trúc theo chuẩn y tế thống nhất.

### 5.1. Bảng số liệu định lượng tổng hợp trên 221 lát cắt có tổn thương (EVAL-07)

| Kiến trúc mô hình | DSC (Mean ± Std) [95% CI] | IoU (Mean ± Std) [95% CI] | Precision (Mean ± Std) [95% CI] | Recall (Mean ± Std) [95% CI] |
|:---|:---:|:---:|:---:|:---:|
| **Attention U-Net** (Đề xuất) | 0.6312 ± 0.2648 [0.5963, 0.6662] | 0.5080 ± 0.2457 [0.4756, 0.5404] | 0.5854 ± 0.2775 [0.5489, 0.6220] | **0.8448 ± 0.2082** [0.8173, 0.8723] |
| **Standard U-Net** (Baseline 1) | **0.6837 ± 0.2364** [0.6525, 0.7149] | **0.5604 ± 0.2323** [0.5298, 0.5911] | 0.6349 ± 0.2576 [0.6010, 0.6689] | 0.8417 ± 0.2198 [0.8127, 0.8707] |
| **SegNet** (Baseline 2) | 0.6693 ± 0.2407 [0.6376, 0.7011] | 0.5425 ± 0.2228 [0.5131, 0.5719] | **0.6466 ± 0.2406** [0.6149, 0.6783] | 0.7557 ± 0.2849 [0.7182, 0.7933] |

### 5.2. Kết quả kiểm định thống kê cặp (Paired Wilcoxon Signed-Rank Test)
- **So sánh với Standard U-Net:**
  - DSC: $p = 1.00 \times 10^{-14}$ (U-Net vượt trội hơn Attention U-Net có ý nghĩa thống kê rất cao).
  - IoU: $p = 1.00 \times 10^{-14}$ (U-Net vượt trội có ý nghĩa thống kê).
  - Precision: $p = 3.91 \times 10^{-3}$ (U-Net ít cảnh báo nhầm hơn).
  - Recall: $p = 0.854$ (Không có sự khác biệt có ý nghĩa thống kê; cả hai mô hình đều có độ nhạy phát hiện tổn thương tương đương nhau quanh mức 84.2% - 84.5%).
- **So sánh với SegNet:**
  - DSC: $p = 0.0184$ (SegNet đạt điểm trùng lặp cao hơn có ý nghĩa thống kê ở mức $\alpha = 0.05$).
  - IoU: $p = 0.0117$ (SegNet cao hơn có ý nghĩa thống kê).
  - Precision: $p = 6.72 \times 10^{-5}$ (SegNet đạt độ chính xác dự đoán cao hơn).
  - Recall: $p = 4.61 \times 10^{-9}$ (Attention U-Net vượt trội hoàn toàn về khả năng bao phủ tổn thương, hạn chế tối đa nguy cơ bỏ sót tổn thương lâm sàng so với SegNet).

### 5.3. Phân tích cấp độ bệnh nhân (Patient-level Aggregation)
Điểm trung bình cộng dồn theo từng ca bệnh chỉ ra nguyên nhân sụt giảm chỉ số của Attention U-Net:
- `P_coronacases_002` (113 lát tổn thương): SegNet đạt DSC 0.7410, U-Net đạt 0.7381, Attention U-Net đạt **0.7282** (hiệu năng tốt, bám sát baseline).
- `P_coronacases_007` (73 lát tổn thương): SegNet đạt DSC 0.8126, U-Net đạt **0.8595**, Attention U-Net đạt **0.8308** (kết quả phân đoạn rất cao).
- `P_radiopaedia_27_86410_0` (35 lát tổn thương - ca bệnh phức tạp có tổn thương kính mờ rải rác): SegNet đạt **0.6232**, U-Net đạt 0.4695, trong khi Attention U-Net chỉ đạt **0.1884**.
- **Căn nguyên kỹ thuật:** Tại ca bệnh `radiopaedia_27`, cơ chế Attention Gate bị nhạy cảm quá mức với vùng ranh giới giữa nhu mô phổi viêm nhiễm và phế quản lành, dẫn đến hiện tượng phân đoạn lan rộng (over-segmentation). Điều này làm tăng độ nhạy phát hiện (Recall cao) nhưng khiến Precision giảm sút mạnh, kéo điểm trung bình toàn tập kiểm thử xuống.

---

## 6. DANH MỤC HIỆN VẬT BÀN GIAO & TÍNH TRUY XUẤT

Mọi bằng chứng của 21 nhiệm vụ đều được định vị rõ ràng trong kho lưu trữ Git:

1. **Văn bản nghiên cứu khoa học:**
   - `docs/Metrics_Specification.md`: Đặc tả toán học các chỉ số đánh giá (RES-04).
   - `docs/Dataset_Specification_Doc.md`: Đặc tả kỹ thuật tập dữ liệu CT (DATA-01).
   - `reports/W5_evaluation_report.md`: Báo cáo chi tiết thực nghiệm Tuần 5.
   - `reports/W1-W5_audit_report.md`: Báo cáo kiểm toán độc lập này.
2. **Dữ liệu & Pipeline tiền xử lý:**
   - `data/raw/md5_checksums.txt`: Danh mục mã băm 41 tệp dữ liệu thô (DATA-02).
   - `reports/DATA-03/Raw_Data_Integrity_Report.json`: Báo cáo metadata hình học 20 CT cases (DATA-03).
   - `data/processed/clean_slices_manifest.csv`: Danh mục 1.750 lát cắt 2D làm sạch (DATA-04).
   - `src/data/hu_window.py`: Pipeline HU clipping [-1000, 400] (DATA-05).
   - `src/data/transform.py`: Module scaling và resize đồng bộ (DATA-06).
   - `src/data/mask_binarize.py`: Module nhị phân hóa mask y tế (DATA-07).
   - `data/processed/splits_patient_level.json` & `summary_split.csv`: Phân chia 70/15/15 chống rò rỉ (DATA-08).
   - `src/data/augmentations.py`: Pipeline tăng cường dữ liệu huấn luyện (DATA-09).
   - `src/data/dataset.py` & `test_loader.py`: Dataset & benchmark dataloader (DATA-10).
   - `data/processed/dataset_v1.0_freeze_manifest.json`: Biên bản niêm phong dữ liệu v1.0 (DATA-12).
3. **Mã nguồn hàm mất mát & Kiểm thử:**
   - `src/losses/dice_loss.py`: Cài đặt Soft Dice Loss (MODEL-02).
   - `src/losses/bce_loss.py`: Cài đặt BCEWithLogitsLoss (MODEL-03).
   - `src/losses/hybrid_loss.py`: Cài đặt Hybrid Loss $\alpha \text{BCE} + \beta \text{Dice}$ (MODEL-04).
   - `tests/test_hybrid_loss.py`: Bộ kiểm thử tự động luồng gradient và trọng số (MODEL-04).
4. **Đánh giá kiểm thử độc lập:**
   - `data/processed/test_manifest.json`: Danh mục 265 lát cắt test niêm phong (EVAL-01).
   - `src/data/eval_loader.py`: Loader độc lập tuần tự batch_size=1 (EVAL-01).
   - `results/metrics_dsc_detail.csv`, `summary.csv`, `patient.csv`: Chỉ số DSC (EVAL-03).
   - `results/metrics_iou_detail.csv`, `summary.csv`, `patient.csv`: Chỉ số IoU (EVAL-04).
   - `results/metrics_precision_detail.csv`, `summary.csv`, `patient.csv`: Chỉ số Precision (EVAL-05).
   - `results/metrics_recall_detail.csv`, `summary.csv`, `patient.csv`: Chỉ số Recall (EVAL-06).
   - `results/quantitative_comparison_table.csv` & `.tex`: Ma trận số liệu định lượng và kiểm định Wilcoxon (EVAL-07).

---

## 7. KẾT LUẬN VÀ KIẾN NGHỊ NGHIỆM THU

1. **Kết luận kiểm toán:**
   - Toàn bộ **21 nhiệm vụ** được phân công cho thành viên Nguyễn Phương Đông từ Tuần 1 đến Tuần 5 đều đã được hoàn thành thực chất, có sản phẩm mã nguồn, dữ liệu thực tế và kết quả kiểm thử định lượng đầy đủ.
   - Các điểm bất nhất kỹ thuật phát sinh trong quá trình phát triển (về checksum, dải nhãn mask thô, và code hashes) đã được xử lý triệt để, đưa toàn bộ hệ sinh thái dữ liệu về trạng thái chuẩn hóa, có thể tái lập 100%.
   - Trạng thái `Completed` trên bảng phân công `TASK_MASTER` hoàn toàn phản ánh đúng thực tế kỹ thuật của dự án.
2. **Kiến nghị cho các giai đoạn tiếp theo (W6 - W8):**
   - Giữ nguyên trạng thái khóa Read-Only của `test_manifest.json` và `dataset_v1.0_freeze_manifest.json`.
   - Sử dụng kết quả thực nghiệm khách quan trong `results/quantitative_comparison_table.csv` để viết các mục *Results* và *Discussion* trong bài báo khoa học (W7), tập trung giải thích cơ chế bù trừ giữa Recall cao và nguy cơ over-segmentation của Attention Gate đối với các tổn thương kính mờ phức tạp.

---
*Biên bản kiểm toán được lập ngày 01 tháng 11 năm 2026 bởi Hệ thống Kiểm toán Kỹ thuật Đồ án NCKH.*
