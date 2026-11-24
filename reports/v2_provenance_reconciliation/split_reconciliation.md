# Báo Cáo Đối Chiếu Phân Chia Bệnh Nhân (Patient Split Reconciliation)

**Thư mục lưu trữ:** `reports/v2_provenance_reconciliation/split_reconciliation.md`  
**Ngày thực hiện:** 24/11/2026  
**Chế độ thực thi:** STRICT READ-ONLY RECONCILIATION  
**Trạng thái đối chiếu:** **PASS (ĐÃ ĐỐI CHIẾU VÀ ĐÍNH CHÍNH DANH SÁCH TEST SET)** ✅  

---

## 1. Mục Tiêu Kiểm Tra

Kiểm tra trực tiếp file cấu hình phân chia bệnh nhân:
`data/processed/splits_patient_level.json`

Nhiệm vụ:
1. Đọc danh sách bệnh nhân chính xác trong từng tập: Train, Validation, Test.
2. Thống kê số lượng bệnh nhân và số lát cắt thuộc 2 nguồn Coronacases và Radiopaedia.
3. Tính toán mã SHA-256 của file split và đối chiếu với trường `split_sha256` trong cả 3 checkpoint.
4. **Xác minh đặc biệt:** Kiểm tra xem danh sách bệnh nhân tập Test có thực sự là `coronacases_006`, `coronacases_008`, `coronacases_010` như một số tài liệu suy đoán trước đó hay không.

---

## 2. Mã Băm Mật Mã và Tính Toàn Vẹn Của File Split

- **Đường dẫn file:** `data/processed/splits_patient_level.json`
- **Mã SHA-256 trên Windows (CRLF endings):**  
  `1c0ddfe4f4407683407c489e5d8904e8c4d53abb7c3bd3e6e20aacf79dcc4010`
- **Mã SHA-256 trên Linux container / Kaggle (LF endings):**  
  `bd1afe5325603baaa7ae4d5ee662f9d6719403b35e585b99087c3c470c7f6f07`
- **Mã `split_sha256` nhúng trong cả 3 checkpoint V2:**  
  `bd1afe5325603baaa7ae4d5ee662f9d6719403b35e585b99087c3c470c7f6f07`

**Kết luận hash:** **TRÙNG KHỚP 100%**. Checkpoint được huấn luyện trên đúng file phân chia bệnh nhân này.

---

## 3. Thống Kê Phân Bổ Bệnh Nhân Giữa Các Tập (Patient-Level Split)

Phân chia tuân thủ tỷ lệ xấp xỉ 70% Train / 15% Validation / 15% Test với hạt giống ngẫu nhiên `seed: 2026`:

| Tập Dữ Liệu | Số Bệnh Nhân | Số Bệnh Nhân Coronacases | Số Bệnh Nhân Radiopaedia | Số Lát Cắt (Slices) | Tỷ Lệ Lát Cắt |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | **14** | 7 | 7 | 1,277 | 72.97% |
| **Validation Set** | **3** | 1 | 2 | 208 | 11.89% |
| **Test Set** | **3** | 2 | 1 | 265 | 15.14% |
| **TỔNG CỘNG** | **20** | **10** | **10** | **1,750** | **100.0%** |

### Chi tiết danh sách bệnh nhân từng tập:

### 3.1. Tập Huấn Luyện (Train Set - 14 bệnh nhân)
- **7 bệnh nhân Coronacases:** `P_coronacases_001`, `P_coronacases_003`, `P_coronacases_004`, `P_coronacases_006`, `P_coronacases_008`, `P_coronacases_009`, `P_coronacases_010`.
- **7 bệnh nhân Radiopaedia:** `P_radiopaedia_10_85902_1`, `P_radiopaedia_10_85902_3`, `P_radiopaedia_14_85914_0`, `P_radiopaedia_29_86490_1`, `P_radiopaedia_29_86491_1`, `P_radiopaedia_4_85506_1`, `P_radiopaedia_7_85703_0`.

### 3.2. Tập Kiểm Định (Validation Set - 3 bệnh nhân)
- **1 bệnh nhân Coronacases:** `P_coronacases_005`.
- **2 bệnh nhân Radiopaedia:** `P_radiopaedia_36_86526_0`, `P_radiopaedia_40_86625_0`.

### 3.3. Tập Kiểm Thử Độc Lập (Independent Test Set - 3 bệnh nhân)
- **2 bệnh nhân Coronacases:** `P_coronacases_002`, `P_coronacases_007`.
- **1 bệnh nhân Radiopaedia:** `P_radiopaedia_27_86410_0`.

---

## 4. Xác Minh Nghiêm Ngặt Về Danh Sách Tập Test (Critical Test Set Verification)

### Câu hỏi thẩm định:
> *Danh sách bệnh nhân Test Set có thực sự là `coronacases_006`, `coronacases_008`, `coronacases_010` hay không?*

### Kết luận xác minh thực tế:
**TUYỆT ĐỐI KHÔNG PHẢI!**

1. **Vị trí thực tế của `coronacases_006`, `coronacases_008`, `coronacases_010`:**
   Cả ba bệnh nhân này nằm hoàn toàn trong **TẬP HUẤN LUYỆN (TRAIN SET)** (`split_data['patient_splits']['train']`).
   - `P_coronacases_006` có 111 lát cắt trong tập Train.
   - `P_coronacases_008` có 110 lát cắt trong tập Train.
   - `P_coronacases_010` có 111 lát cắt trong tập Train.

2. **Hậu quả nếu sử dụng danh sách này làm Test Set:**
   Nếu đánh giá mô hình trên `coronacases_006`, `008`, `010`, đó sẽ là hành vi **đánh giá trên tập huấn luyện (Evaluating on Training Data)** — một lỗi rò rỉ dữ liệu (Data Leakage) nghiêm trọng làm sai lệch toàn bộ kết quả nghiên cứu.

3. **Danh sách Test Set ĐÍCH THỰC (True Independent Test Set):**
   Chính xác 3 bệnh nhân chưa từng xuất hiện trong quá trình tối ưu hóa trọng số:
   - **`P_coronacases_002`** (111 lát cắt, 32 lát có tổn thương).
   - **`P_coronacases_007`** (112 lát cắt, 20 lát có tổn thương).
   - **`P_radiopaedia_27_86410_0`** (42 lát cắt, 20 lát có tổn thương).
   - **Tổng cộng tập Test:** **265 lát cắt** (72 lát có tổn thương, 193 lát lành).

*Ghi chú nguồn gốc nhầm lẫn:* Việc ghi nhận `coronacases_006`, `008`, `010` trong văn bản trước đó là lỗi giả định khi soạn thảo văn bản, không xuất phát từ file split hay mã nguồn. Bản báo cáo này chính thức bác bỏ và đính chính danh sách Test Set.

---

## 5. Kết Luận Đối Chiếu Phân Chia Bệnh Nhân

| Tiêu Chí Thẩm Định | Kết Quả Thực Tế | Phân Loại |
| :--- | :--- | :--- |
| **Mã hash Split SHA-256** | Trùng khớp 100% với `split_sha256` trong cả 3 checkpoint (`bd1afe...`). | **PASS** |
| **Không rò rỉ dữ liệu (No Leakage)** | Tập Test (`P_002`, `P_007`, `P_rad_27`) hoàn toàn độc lập với Train và Val. | **PASS** |
| **Đính chính Test Set** | Đã xác minh bác bỏ nhầm lẫn về `coronacases_006, 008, 010`. | **PASS** |

**KẾT LUẬN CUỐI CÙNG: PASS ✅** (Phân chia bệnh nhân hoàn toàn minh bạch, bảo toàn tính toàn vẹn khoa học).
