# Danh sách thông tin còn thiếu — Bài báo Attention U-Net (v15)

Tài liệu gửi dev để kiểm tra lại. Phạm vi: 4 chỗ tô vàng trong `Bai_bao_Attention_UNet_v15.docx` và các điểm không tô vàng nhưng mâu thuẫn với bằng chứng đã có.

Nguyên tắc: không đưa số liệu vào bài nếu chưa xác định được checkpoint, cấu hình và code đã tạo ra chúng.

---

## A. Bốn chỗ đang tô vàng

### A1. Mục 2.2 (Tiền xử lý): cửa sổ HU áp dụng cho nhóm radiopaedia

**Nội dung hiện tại:** nhóm radiopaedia (10 ca) có cường độ 0–255, không theo thang HU, nhưng vẫn bị áp cửa sổ [−1000, 400] HU chung với nhóm coronacases. Câu này được tô vàng vì chưa có quyết định.

**Đã biết:**
- Coronacases: khoảng −1023 đến 9567 HU.
- Radiopaedia: khoảng 0–255 (ví dụ radiopaedia_27 có khoảng [4, 255]).
- Ca radiopaedia_27_86410_0 nằm trong tập kiểm thử (37 lát).

**Còn thiếu:**
1. Quyết định: giữ như một hạn chế của nghiên cứu, hay xử lý lại nhóm radiopaedia (ví dụ chuẩn hóa min–max riêng cho từng ca) và huấn luyện lại.
2. Nếu xử lý lại: công thức chuẩn hóa cụ thể, và xác nhận bằng code thực tế.
3. Nếu giữ hạn chế: xác nhận đoạn văn hạn chế ở mục Kết luận đã mô tả đúng.

**Ai cung cấp:** dev (quyết định); người viết (sửa văn bản).

---

### A2. Mục 2.3 (Huấn luyện): seed 2026 "xác nhận các checkpoint mới được huấn luyện với seed này"

**Nội dung hiện tại:** văn bản khẳng định checkpoint được huấn luyện với seed 2026, dựa trên hàm `set_seed` trong `src/utils/seed.py`.

**Đã kiểm tra:**
- Log của Version 1 (notebook "final version", ngày 31/10) không có dòng "Random Seed locked to: 2026". Ctrl+F trả về 0 kết quả, và dòng 25–50 của log cũng không có.
- Chưa kiểm tra xem chuỗi "Random Seed locked" có tồn tại trong code hay không.

**Còn thiếu:**
1. Log huấn luyện của đúng checkpoint được dùng cho bài báo, có dòng seed (nguyên văn, kèm đường dẫn file).
2. Hoặc: kết quả `git log -S "Random Seed locked"` trong repo, cho thấy dòng in seed có tồn tại trong code.
3. Hoặc: nếu không có log, chỉ được viết "cấu hình seed = 2026", không được viết "log xác nhận".

**Ai cung cấp:** dev, hoặc người chạy run gốc.

---

### A3. Mục 3.2 (Kết quả): liên kết file dự đoán với checkpoint EXP-ATTUNET-001_best.pth

**Nội dung hiện tại:** văn bản nói chưa xác nhận liên kết giữa ba file CSV/dự đoán và checkpoint. Checkpoint local chỉ có `best_val_dice = 0,901` và không lưu cấu hình, seed hay commit.

**Đã biết:**
- `predictions_manifest.json` của att_unet ghi checkpoint `/kaggle/working/lung-lesion-segmentation/checkpoints/EXP-ATTUNET-001_best.pth`, kích thước 359,42 MB, thời điểm 2026-11-04 06:33:54 UTC (13:33 giờ Việt Nam), thiết bị cuda, 265 mẫu, ngưỡng 0,5.
- Checkpoint local có kích thước khoảng 125,67 MB (chỉ có trọng số). File Kaggle gấp khoảng 3 lần, khớp với việc có thêm trạng thái optimizer. Đây là suy luận từ kích thước, chưa được xác nhận.
- Version 1 của notebook (huấn luyện 31/10) có Val Dice 1.0 không hợp lệ, và CSV của nó có toàn bộ `tn=65536`, nghĩa là mặt nạ mục tiêu rỗng. Đây là lỗi của run đó, không phải nguồn của bài báo.
- Ba file CSV đã gửi (Attn, U-Net, SegNet) có DSC khác nhau, nhưng chưa xác định được chúng được tạo từ run nào.
- Predictions của U-Net và SegNet không có manifest.

**Còn thiếu:**
1. Phiên Kaggle đã tạo predictions ngày 4/11 (tên notebook, tài khoản, phiên bản). Hiện chưa tìm được phiên bản nào khớp; các version 3, 4, 5 chỉ là xuất HTML.
2. sha256 và thời điểm sửa đổi của file `EXP-ATTUNET-001_best.pth` từ phiên đó.
3. Kết quả so sánh: chạy checkpoint trên một lát test và so với file `.npy` đã lưu; sai số tối đa phải dưới khoảng 1e-4 (chấp nhận được với khác biệt cuda/cpu).
4. Manifest predictions cho U-Net và SegNet.
5. Xác nhận checkpoint này không phải của Version 1 (Val Dice 1.0).

**Ai cung cấp:** dev, hoặc người có quyền truy cập phiên Kaggle ngày 4/11.

---

### A4. Hình 3: ảnh chụp màn hình giao diện demo

**Nội dung hiện tại:** placeholder "Chèn ảnh chụp màn hình giao diện demo".

**Đã biết:**
- Lần chụp trước dùng file `radiopaedia_5_0.nii.gz` và cho kết quả không hợp lý: panel phổi gần như đen, vùng tổn thương 100%. Ảnh minh họa của bạn chỉ có đen trắng, khớp với mặt nạ `Infection_Mask`, không phải CT. Đây là giả thuyết cần xác nhận bằng cách kiểm tra khoảng giá trị của file.
- Ca coronacases_010, lát 150 cũng cho panel phổi gần đen và gánh nặng 100%.

**Còn thiếu:**
1. Xác nhận file tải lên là CT (thư mục `COVID-19-CT-Seg_20cases`), không phải mặt nạ (thư mục `Infection_Mask`). Lệnh kiểm tra: giá trị CT có dải rộng, mặt nạ chỉ có 0 và 1.
2. Tên file CT, chỉ số lát, và cài đặt demo (cửa sổ, ngưỡng 0,5).
3. Ảnh chụp phải cho thấy phổi rõ ràng và gánh nặng hợp lý. Nếu không đạt, không dùng.
4. Chú thích hình phải ghi tên checkpoint. Nếu A3 chưa xác nhận, chú thích phải ghi rõ là ảnh minh họa, không phải kết quả nghiên cứu.
5. Nếu dùng ảnh JPEG bên ngoài: phải ghi nguồn, và chỉ dùng để minh họa giao diện.

**Ai cung cấp:** người chạy demo; dev xác nhận checkpoint.

---

## B. Điểm KHÔNG tô vàng nhưng cần dev kiểm tra

Các câu sau đang được trình bày như đúng nhưng chưa được đối chiếu với bằng chứng gần đây. Cần kiểm tra trước khi nộp.

| # | Vị trí | Nội dung hiện tại | Vấn đề | Cần kiểm tra |
|---|---|---|---|---|
| B1 | 2.3 | "Số epoch tối đa là 100" | Log của run Attention U-Net ghi `Epochs: 50`. | Số epoch thực tế của checkpoint được dùng. |
| B2 | 2.3 | "SegNet được huấn luyện bằng script riêng với cấu hình khác" (Bảng 1) | Kaggle ghi 50 epoch, config ghi 100. Checkpoint SegNet local có best_val_dice khác. | Cấu hình SegNet thực tế: epoch, batch, optimizer, weight decay, lịch học. |
| B3 | 2.3 | "Các file configs/attention_unet.yaml, baseline_unet.yaml và segnet.yaml" | Chưa kiểm tra tên file và giá trị. | Mở từng file, xác nhận giá trị khớp Bảng 1. |
| B4 | 3.1 / 3.2 | Bảng 3, Bảng 4, DSC 0,9791 / 0,9740 / 0,9662, p-values | Số liệu được tính từ ba CSV, nhưng nguồn của CSV chưa xác định (xem A3). | Chạy lại `check_csv.py` trên ba file đã có, và xác nhận nguồn (A3). |
| B5 | 3.2 | "Recall của Attention U-Net là 0,9941" | Chưa kiểm tra lại. | Tính lại từ CSV. |
| B6 | 2.2 | "các lát không đạt ngưỡng diện tích phổi (≥ 3.000 pixel) hoặc tổn thương (≥ 15 pixel) bị loại bỏ" | Ngưỡng này phải khớp với `clean_slices_manifest.csv` và code tiền xử lý. | Đối chiếu code và manifest (1750 dòng). |
| B7 | 4 (Hạn chế) | "SegNet dùng 32 kênh cơ bản" | Chưa kiểm tra trong code. | Xác nhận trong `src/models/segnet.py` hoặc config. |
| B8 | 2.1 | Giấy phép dữ liệu: CC BY 4.0 (Zenodo) | README trên Kaggle ghi CC BY-NC-SA. | Quyết định cách ghi, hoặc xác nhận giấy phép đúng. |
| B9 | Tóm tắt | Các số DSC trong tóm tắt | Phải khớp với Bảng 3 và CSV. | Đối chiếu sau khi có kết quả B4. |

---

## C. Các phát hiện từ phiên huấn luyện Version 1 (không dùng cho bài báo)

- Cả ba run (Attn, U-Net, SegNet) trong notebook "final version" đều có Val Dice 1.0 từ epoch 2–5 và giữ nguyên. Checkpoint "tốt nhất" được chọn tại epoch đầu tiên đạt 1.0.
- Bảng đánh giá cuối trong notebook: DSC, IoU, Precision, Recall = 1.0000 ± 0.0000 cho cả hai chế độ (221 và 265 lát).
- CSV `test_metrics_slice_level.csv` của Version 1: toàn bộ `tp=fp=fn=0`, `tn=65536`, `dice=1.0`, kể cả các dòng có `lesion_present=1`. Mặt nạ mục tiêu trong pipeline rỗng.
- Hàm Dice đã kiểm tra độc lập và đúng (mặt nạ với chính nó = 1; với toàn 0 ≈ 0; lệch 20 px = 0,36).
- Nguyên nhân có khả năng: nhãn không khớp (`lesion_label` có thể là 3). Cần xác nhận bằng dòng `lesion_label` trong config của Version 1 (thư mục `lung-lesion-segmentation/configs` trong tab Output).
- Hệ quả: mọi số liệu, log và checkpoint của Version 1 không được dùng trong bài báo.

---

## D. Thứ tự đề xuất để dev xử lý

1. Xác nhận `lesion_label` trong config của Version 1 (C).
2. Xác định nguồn của ba CSV trong bài báo (A3, B4). Đây là bước quan trọng nhất, vì toàn bộ kết quả phụ thuộc vào nó.
3. Xác nhận checkpoint và kiểm tra khớp predictions (A3).
4. Tìm log có seed (A2) và cấu hình SegNet (B2).
5. Quyết định về radiopaedia 0–255 (A1).
6. Chụp lại Hình 3 bằng CT thật sau khi có checkpoint đã xác nhận (A4).
7. Nếu nguồn CSV không xác định được, cần huấn luyện và đánh giá lại với pipeline đúng, có log và seed đầy đủ.
