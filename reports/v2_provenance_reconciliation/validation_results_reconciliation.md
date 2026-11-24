# Báo Cáo Đối Chiếu Kết Quả Validation và Lịch Sử Huấn Luyện (Validation Results Reconciliation)

**Thư mục lưu trữ:** `reports/v2_provenance_reconciliation/validation_results_reconciliation.md`  
**Ngày thực hiện:** 24/11/2026  
**Chế độ thực thi:** STRICT READ-ONLY RECONCILIATION  
**Trạng thái đối chiếu:** **PASS** ✅  

---

## 1. Mục Tiêu Kiểm Tra

1. Trích xuất chính xác siêu dữ liệu và các chỉ số validation từ 3 checkpoint `.pth` trong `D:\TTNT\v2\checkpoints\`.
2. Đối chiếu với các file nhật ký huấn luyện tương ứng tại `D:\TTNT\v2\logs\`.
3. Làm rõ nguồn gốc và bản chất của sự khác biệt giữa các con số từng được báo cáo:
   - **Attention U-Net:** `0.7627` vs `0.781709`
   - **Standard U-Net:** `0.7823` vs `0.768502`
   - **SegNet:** `0.760679`

---

## 2. Dữ Liệu Trích Xuất Trực Tiếp Từ Checkpoint V2 Hiện Hành

Dữ liệu được trích xuất bằng Python trực tiếp từ các file `.pth` trong `D:\TTNT\v2\checkpoints\`:

| Trường Siêu Dữ Liệu | Attention U-Net (`EXP-ATTUNET-001`) | Standard U-Net (`EXP-UNET-001`) | SegNet (`EXP-SEGNET-001`) |
| :--- | :--- | :--- | :--- |
| **Tên file Checkpoint** | `EXP-ATTUNET-001_best.pth` | `EXP-UNET-001_best.pth` | `EXP-SEGNET-001_best.pth` |
| **Dung lượng file (Bytes)**| 376,881,899 bytes | 372,584,491 bytes | 28,262,777 bytes |
| **Mã SHA-256 Checkpoint** | `fef6d0e3641cf1a55611166...` | `a0101e39d6b8f0183eb4a9...` | `9745790624e7e5caaaae1b...` |
| **Lớp mô hình (`model_name`)**| `AttentionUNet` | `UNet` | `SegNet` |
| **Epoch tối ưu (`epoch`)**| **Epoch 20** | **Epoch 34** | **Epoch 20** |
| **Best Val Dice (`best_val_dice`)**| **0.781708807493489** | **0.7685020974696599** | **0.7606793421806348** |
| **Val Loss (Hybrid Loss)** | 0.20109222313532463 | 0.18605836595480257 | 0.18295476900843474 |
| **Val IoU (Jaccard Index)** | 0.6815718089467784 | 0.6714003447282286 | 0.6587076484028221 |
| **Val Precision** | 0.8544485345753863 | 0.8652960077051980 | 0.8131291324272097 |
| **Val Recall** | 0.7846272526431916 | 0.7732485130783276 | 0.7879413432508475 |
| **Git Commit Hash** | `34f00e06ef0a410f45f3e254...` | `34f00e06ef0a410f45f3e254...` | `34f00e06ef0a410f45f3e254...` |
| **Data Manifest SHA-256** | `7e3f6dadfef2862f5d0306fc...` | `7e3f6dadfef2862f5d0306fc...` | `7e3f6dadfef2862f5d0306fc...` |
| **Split SHA-256** | `bd1afe5325603baaa7ae4d5e...` | `bd1afe5325603baaa7ae4d5e...` | `bd1afe5325603baaa7ae4d5e...` |
| **Preprocessing Version** | `v2_dual_source` | `v2_dual_source` | `v2_dual_source` |
| **Random Seed** | `2026` | `2026` | `2026` |
| **Lesion Label** | `1` | `1` | `1` |

---

## 3. Đối Chiếu Với Nhật Ký Huấn Luyện (Training Logs)

### 3.1. Đối chiếu SegNet (`EXP-SEGNET-001_training.log`)
File `D:\TTNT\v2\logs\EXP-SEGNET-001_training.log` ghi lại đầy đủ 35 epochs:
- Tại **Epoch 20**:
  `20,0.160076,0.182955,0.760679,0.658708,0.813129,0.787941,0.00009055,28.76`
  Trùng khớp 100% với `val_metrics` trong `EXP-SEGNET-001_best.pth`.
- Tại **Epoch 35** (sau 15 epochs không tăng, kích hoạt Early Stopping):
  `35,0.145529,0.172067,0.745551,0.645541,0.765247,0.833441,0.00007297,29.06`
  Trùng khớp 100% với `EXP-SEGNET-001_last.pth`.

### 3.2. Đối chiếu Attention U-Net và Standard U-Net
- Nhật ký log dạng text của AttUNet (23 dòng) và UNet (25 dòng) bị gián đoạn ghi chép do hàm `open(log_file, "w")` mở ghi mới khi kernel Kaggle bị refresh/restart.
- Tuy nhiên, bên trong file checkpoint `.pth`, đối tượng `val_metrics` và `best_val_dice` được lưu trữ trực tiếp bằng `torch.save()`, bảo toàn nguyên vẹn toàn bộ 5 chỉ số đo lường tại thời điểm lưu trọng số tối ưu.

---

## 4. Giải Trình Nguồn Gốc và Phân Định Hai Lần Chạy Thực Nghiệm (Provenance Reconciliation)

Kiểm tra nhật ký đối thoại và lịch sử console Kaggle cho thấy:

### 4.1. Lần chạy thứ nhất (Run 1 - Kaggle Session Đầu):
- **Bối cảnh:** Huấn luyện trên Kaggle trong phiên làm việc ban đầu.
- **Kết quả ghi nhận trên console:**
  - Attention U-Net kích hoạt Early Stopping: **Val Dice = 0.7627** (76.27%).
  - Standard U-Net hoàn thành sau đó: **Val Dice = 0.7823** (78.23%).
- **Nguyên nhân không lưu trữ checkpoint này:** Phiên làm việc của Kaggle gặp lỗi đồng bộ draft (`ConcurrencyViolation Sequence number must match Draft record`) hoặc bị hết thời gian session draft, buộc người dùng phải refresh và khởi động lại quá trình chạy.

### 4.2. Lần chạy thứ hai (Run 2 - Final V2 Run, Checkpoint Hiện Tại):
- **Bối cảnh:** Phiên làm việc mới sau khi khởi tạo lại môi trường và thiết lập phân chia ổn định.
- **Kết quả thực tế ghi nhận và lưu trữ trong file `.pth`:**
  - **Attention U-Net:** Đạt đỉnh tại Epoch 20 với **Val Dice = 0.781709** (78.17%), dừng tại Epoch 35. Checkpoint: `EXP-ATTUNET-001_best.pth`.
  - **Standard U-Net:** Đạt đỉnh tại Epoch 34 với **Val Dice = 0.768502** (76.85%), dừng tại Epoch 39. Checkpoint: `EXP-UNET-001_best.pth`.
  - **SegNet:** Đạt đỉnh tại Epoch 20 với **Val Dice = 0.760679** (76.07%), dừng tại Epoch 35. Checkpoint: `EXP-SEGNET-001_best.pth`.
- **Tình trạng đóng gói:** Toàn bộ 6 file checkpoint thuộc Lần chạy thứ hai này đã được nén vào `checkpoints_att_and_base.zip` và `segnet_models_checkpoints.zip`, sau đó giải nén vào thư mục `D:\TTNT\v2\checkpoints\`.

### 4.3. Kết luận phân định:
- Cả hai bộ số liệu (`0.7627 / 0.7823` và `0.781709 / 0.768502 / 0.760679`) đều phản ánh đúng quá trình hội tụ của mô hình trong khoảng Dice 76% - 78% trên cùng tập Validation.
- Tuy nhiên, **chỉ có Run 2 mới có file checkpoint `.pth` vật lý được lưu trữ và kiểm toán đầy đủ**.
- Do đó, để đảm bảo tính tái lập và kiểm chứng thực nghiệm độc lập, **toàn bộ quá trình Test Evaluation tiếp theo sẽ sử dụng các checkpoint vật lý hiện có của Run 2 (`0.781709`, `0.768502`, `0.760679`)**. Báo cáo này không xóa bỏ số liệu cũ mà xác lập rõ nguồn gốc phân kỳ.

---

## 5. Kết Luận Đối Chiếu Kết Quả Validation

| Hạng Mục | Kết Quả Phân Tích | Đánh Giá |
| :--- | :--- | :--- |
| **Siêu dữ liệu trong Checkpoint** | Đầy đủ 100%, không khuyết thiếu trường nào. | **PASS** |
| **Đối chiếu SegNet Log vs Checkpoint**| Trùng khớp tuyệt đối 100% tại Epoch 20 và Epoch 35. | **PASS** |
| **Làm rõ sự khác biệt giữa các lần chạy**| Xác định rõ Run 1 (console draft cũ) và Run 2 (checkpoint vật lý hiện hành). | **PASS** |

**KẾT LUẬN CUỐI CÙNG: PASS ✅**
