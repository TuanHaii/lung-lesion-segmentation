# TECH NOTE: PHÂN TÍCH VÀ ĐỐI CHUẨN KIẾN TRÚC U-NET, ATTENTION U-NET VÀ SEGNET

* **Mã công việc:** `RES-03` — Phân tích các kiến trúc phân đoạn được lựa chọn
* **Thuộc đề tài:** Nghiên cứu và ứng dụng kiến trúc U-Net kết hợp cơ chế Attention trong bài toán phân đoạn tổn thương phổi từ ảnh cắt lớp vi tính (CT Scans)
* **Người thực hiện:** Hải (AI & System Engineering)
* **Tài liệu đầu ra (Deliverable):** `Tech_Note_UNet_Architecture.md`
* **Mục đích:** Cung cấp phân tích kỹ thuật chuyên sâu về cấu trúc Encoder–Decoder, Skip Connections, Pooling/Upsampling và Attention Gate nhằm phục vụ trực tiếp thiết kế thực nghiệm mô hình (W3–W4), giải quyết RQ1 và RQ2.

---

## 1. TỔNG QUAN VÀ BỐI CẢNH THỰC NGHIỆM

Trong bài toán phân đoạn tổn thương phổi (kính mờ – GGO, đông đặc – Consolidation) từ ảnh CT lát cắt 2D:
* **Đặc điểm bài toán:** Tổn thương thường chiếm diện tích nhỏ trên lát cắt, ranh giới mờ, độ tương phản thấp so với mô phế nang xung quanh, và có nguy cơ dương tính giả cao ở thành ngực/xương sườn/mạch máu.
* **Bộ 3 mô hình nghiên cứu:**
  1. **Standard U-Net (Baseline đối chuẩn):** Kiến trúc kinh điển với kết nối tắt trực tiếp ghép nối đặc trưng mức thấp và mức cao.
  2. **Attention U-Net (Mô hình nghiên cứu trọng tâm):** Tích hợp module Attention Gate (AG) vào skip connections nhằm lọc nhiễu không gian và làm nổi bật vùng tổn thương.
  3. **SegNet (Mô hình đối chuẩn mở rộng):** Đại diện cho cơ chế lưu vết vị trí cực đại (Pooling Indices) để khôi phục biên không gian mà không truyền trực tiếp toàn bộ feature maps.

---

## 2. PHÂN TÍCH CHI TIẾT KIẾN TRÚC MÔ HÌNH

### 2.1. Standard U-Net (Ronneberger et al., 2015)
* **Encoder (Contracting Path):** 
  * Gồm 4 tầng giảm mẫu; mỗi tầng là một khối tích chập kép (`DoubleConv`: $3\times3\text{ Conv} \to \text{BatchNorm} \to \text{ReLU} \to 3\times3\text{ Conv} \to \text{BatchNorm} \to \text{ReLU}$).
  * Sau mỗi tầng, không gian ảnh giảm một nửa qua $2\times2\text{ Max-Pooling}$ (stride = 2), số kênh tăng dần: $1 \to 64 \to 128 \to 256 \to 512$.
  * *Vai trò:* Trích xuất ngữ nghĩa toàn cục và bối cảnh bệnh học ("Đây là dạng tổn thương gì?").
* **Bottleneck:**
  * Đáy chữ U nén không gian xuống $16\times16$ (với đầu vào $256\times256$), số kênh đạt mức tối đa ($1024$). Chứa thông tin trừu tượng mức cao nhất.
* **Decoder (Expanding Path):**
  * Gồm 4 tầng tăng mẫu tương ứng; mỗi bước sử dụng $2\times2\text{ Transposed Convolution}$ (Up-conv / ConvTranspose2d, stride = 2) để nhân đôi kích thước không gian và giảm nửa số kênh.
  * *Skip Connection:* Sao chép trực tiếp feature map ở cùng cấp độ từ Encoder sang Decoder và thực hiện phép ghép nối (`Concatenation`) dọc theo trục channels trước khi đưa qua `DoubleConv`.
  * *Tầng phân lớp cuối:* Lớp tích chập $1\times1\text{ Conv}$ ánh xạ số kênh về 1, kết hợp hàm kích hoạt `Sigmoid` đưa xác suất về khoảng $[0.0, 1.0]$.
* **Ưu điểm & Nhược điểm đối với ảnh CT:**
  * *Ưu điểm:* Khôi phục chi tiết biên tốt nhờ truyền trực tiếp đặc trưng độ phân giải cao qua Skip Connections.
  * *Nhược điểm:* Truyền toàn bộ đặc trưng bao gồm cả mô lành, thành ngực, xương sườn gây nhiễu và dễ tạo ra dương tính giả (False Positives).

---

### 2.2. Attention U-Net (Oktay et al., 2018)
* **Cơ chế tổng thể:**
  * Giữ nguyên cấu trúc đối xứng Encoder, Bottleneck và Decoder của Standard U-Net.
  * Cải tiến then chốt: Đặt thêm module **Attention Gate (AG)** trên từng đường truyền skip connection trước khi thực hiện phép ghép nối (`Concatenation`).
* **Đặc tả toán học Attention Gate (AG):**
  * AG nhận 2 tín hiệu đầu vào:
    1. $x^l$: Bản đồ đặc trưng không gian (Spatial Features) từ tầng $l$ của nhánh Encoder.
    2. $g$: Tín hiệu điều khiển (Gating Signal) lấy từ tầng sâu hơn $l+1$ của Decoder (mang ngữ nghĩa bao quát).
  * Công thức tính hệ số chú ý $\alpha$:
    $$q_{att}^l = \psi^T \left( \sigma_1 \left( W_x^T x^l + W_g^T g + b_g \right) \right) + b_\psi$$
    $$\alpha^l = \sigma_2 \left( q_{att}^l \right)$$
    * Trong đó: $W_x, W_g, \psi$ được cài đặt bằng các phép tích chập $1\times1$; $\sigma_1$ là hàm kích hoạt $\text{ReLU}$; $\sigma_2$ là hàm $\text{Sigmoid}$ để đưa trọng số về dải $[0, 1]$.
  * Đầu ra của Attention Gate:
    $$\hat{x}^l = \alpha^l \odot x^l$$
    * Với $\odot$ là phép nhân từng phần tử (element-wise multiplication).
* **Vai trò trong phân đoạn CT phổi (Phục vụ RQ1):**
  * Tín hiệu gating $g$ hướng dẫn việc tái cân bằng trọng số không gian: triệt tiêu (làm mờ về 0) các vùng mô lành, phế nang bình thường và xương sườn.
  * Làm sáng rực các vùng tổn thương thực sự (đặc biệt là tổn thương dạng kính mờ GGO và các nốt u nhỏ), giúp Decoder tập trung tối đa vào vùng bệnh.

---

### 2.3. SegNet (Badrinarayanan et al., 2017)
* **Cấu trúc tổng thể:**
  * Kiến trúc Encoder–Decoder đối xứng dựa trên cấu trúc các khối VGG-16.
* **Cơ chế Max-Pooling Indices & Max-Unpooling:**
  * *Tại Encoder:* Khi thực hiện phép $2\times2\text{ Max-Pooling}$, SegNet lưu lại ma trận chỉ số vị trí của giá trị lớn nhất trong từng cửa sổ $2\times2$ (gọi là `pooling indices` / `argmax`).
  * *Tại Decoder:* SegNet **không dùng Skip Connections** dạng truyền toàn bộ tensor đặc trưng, cũng không dùng Transposed Convolution có trọng số học được. Thay vào đó, nó sử dụng phép **Max-Unpooling**: đưa các giá trị đặc trưng ở tầng dưới lên đúng vị trí tọa độ đã lưu trong `pooling indices`, các vị trí còn lại trong cửa sổ $2\times2$ được điền giá trị 0.
  * Sau đó, feature map thưa thớt này được đưa qua các lớp tích chập chuẩn để làm mịn và phục hồi thông tin.
* **Ưu điểm & Nhược điểm đối với ảnh CT (Phục vụ RQ2):**
  * *Ưu điểm:* Tiết kiệm bộ nhớ đáng kể (chỉ lưu ma trận chỉ số nguyên thay vì lưu toàn bộ tensor số thực 32-bit), duy trì cấu trúc biên của các vật thể nhỏ gọn.
  * *Nhược điểm:* Việc thiếu đường truyền đặc trưng trực tiếp khiến thông tin ngữ nghĩa mức thấp dễ bị thất thoát; vùng dự đoán bên trong tổn thương có thể bị rỗng hoặc không đồng nhất.

---

## 3. BẢNG SO SÁNH ĐỐI CHUẨN KỸ THUẬT PHỤC VỤ THỰC NGHIỆM

| Tiêu chí kỹ thuật | Standard U-Net | Attention U-Net | SegNet |
| :--- | :--- | :--- | :--- |
| **Vai trò nghiên cứu** | Baseline chính | Mô hình nghiên cứu trọng tâm | Baseline đối chứng bổ sung |
| **Cơ chế Downsampling** | $2\times2\text{ Max-Pooling}$ | $2\times2\text{ Max-Pooling}$ | $2\times2\text{ Max-Pooling}$ (lưu `indices`) |
| **Cơ chế Upsampling** | Transposed Conv ($2\times2$, stride 2) | Transposed Conv ($2\times2$, stride 2) | Max-Unpooling (dựa trên `indices`) |
| **Cơ chế liên kết (Skip)** | Ghép nối trực tiếp (`Concat`) | Lọc qua Attention Gate rồi `Concat` | Không có (chỉ truyền `pooling indices`) |
| **Số lượng tham số (Params)** | Trung bình (~31.0M) | Tăng nhẹ (+1.5% đến 2.0% do AG) | Thấp đến trung bình (~29.4M) |
| **Bộ nhớ VRAM (Training)** | Trung bình | Tăng nhẹ (thêm tensor gating) | Tiết kiệm nhất trong 3 mô hình |
| **Độ nhạy biên tổn thương** | Tốt, nhưng dễ nhiễu mô lành | Tốt nhất, lọc nhiễu nền | Tốt ở biên nốt nhỏ, kém ở vùng mờ |
| **Phù hợp với dạng tổn thương** | Tổn thương trung bình & lớn | Tổn thương nhỏ, kính mờ (GGO) | Nốt đơn độc có ranh giới rõ |

---

## 4. ĐỊNH HƯỚNG THIẾT KẾ THỰC NGHIỆM ĐỒNG BỘ (W3 – W5)

Để đảm bảo kết quả so sánh khách quan và đáp ứng các quy chuẩn khoa học (Evidence-based):

1. **Chuẩn hóa giao diện Input/Output:**
   * Cả 3 mô hình nhận chung đầu vào Tensor: `[Batch_Size, 1, 256, 256]`, chuẩn hóa Min-Max $[0.0, 1.0]$ sau khi cắt ngưỡng HU $[-1000, 400]$.
   * Cả 3 mô hình trả về ma trận xác suất nhị phân: `[Batch_Size, 1, 256, 256]` qua hàm `Sigmoid`.

2. **Đồng nhất hàm tổn thất (Hybrid Loss):**
   * Sử dụng thống nhất:
     $$\mathcal{L}_{\text{hybrid}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{SoftDice}}$$
   * Soft Dice Loss bắt buộc có hệ số làm mịn $\epsilon = 10^{-5}$ để tránh lỗi chia cho 0.

3. **Cấu hình siêu tham số (Hyperparameters):**
   * Optimizer: `AdamW` (learning rate ban đầu $10^{-4}$, weight decay $10^{-4}$).
   * Scheduler: `CosineAnnealingLR`.
   * Batch size: 16 (phù hợp VRAM máy trạm), tối đa 80–100 epochs kèm Early Stopping (patience = 15 trên Validation Set).
   * Cùng hạt giống ngẫu nhiên (random seed) cho cả 3 lượt huấn luyện.

4. **Trọng tâm kiểm chứng câu hỏi nghiên cứu:**
   * **Kiểm chứng RQ1 (Attention U-Net vs Standard U-Net):**
     * Đánh giá độ chênh lệch chỉ số DSC và IoU, đặc biệt trên tập con các tổn thương nhỏ (diện tích $< 5\%$ diện tích phổi) và các ca bệnh có tổn thương kính mờ (GGO).
     * Kiểm định ý nghĩa thống kê bằng paired Wilcoxon signed-rank test ($p\text{-value} < 0.05$).
   * **Kiểm chứng RQ2 (U-Net vs SegNet):**
     * So sánh khả năng duy trì độ sắc nét của đường viền biên (Boundary Delineation) qua các phép phóng to (Zoom-in visualization).
     * Đánh giá sự đánh đổi (Trade-off) giữa độ chính xác phân đoạn, số lượng tham số (FLOPs/Params) và độ trễ suy luận (latency ms/slice) phục vụ triển khai Web Demo.