# HƯỚNG DẪN SỬ DỤNG VÀ VẬN HÀNH WEB DEMO (STREAMLIT)
**Dự án: Phân đoạn tổn thương phổi từ ảnh CT bằng Attention U-Net (Lung Lesion Segmentation)**  
*Mã tài liệu: `docs/07_WEB_DEMO_GUIDE.md`*

---

## 1. Tổng quan hệ thống Web Demo

Giao diện Web Demo được xây dựng trên nền tảng **Streamlit**, cung cấp giao diện trực quan hỗ trợ bác sĩ và nhà nghiên cứu thử nghiệm các mô hình AI trong chẩn đoán hình ảnh CT lồng ngực (COVID-19 / Tổn thương viêm phổi).

### Các tính năng cốt lõi:
1. **Chẩn đoán lát cắt đơn (Single Slice Diagnosis):**
   - Hỗ trợ dữ liệu CT mô phỏng 3D y khoa hoặc tải lên file CT thực tế (`.nii`, `.nii.gz`, `.png`, `.jpg`).
   - Tự động điều chỉnh cửa sổ Hounsfield Unit (HU Windowing) theo nhu mô phổi, trung thất, xương.
   - Trực quan hóa 4 khung hình: Ảnh CT gốc, Bản đồ nhiệt xác suất (Probability Heatmap), Lớp phủ nhị phân (Binary Mask Overlay) và Nhãn tham chiếu (Ground Truth).
   - Đo lường định lượng lâm sàng tự động: Diện tích tổn thương ($cm^2$), Tỷ lệ gánh nặng phổi (Burden %), Mức độ nghiêm trọng và Thời gian suy luận (Latency ms).

2. **So sánh trực tiếp đa mô hình (Model Comparison):**
   - Đặt song song 2 mô hình (Attention U-Net đề xuất vs Standard U-Net hoặc SegNet baseline).
   - Minh chứng trực quan khả năng ức chế cảnh báo nhầm (False Positives) của cơ chế Attention Gate.

3. **Bảng số liệu & Đồ thị thực nghiệm (Benchmark & Learning Curves):**
   - Hiển thị bảng đối chuẩn các chỉ số độc lập: Dice Score, IoU (Jaccard), Precision, Recall.
   - Vẽ biểu đồ hội tụ Validation Loss và Validation Dice qua từng Epoch từ log huấn luyện.

---

## 2. Cấu trúc thư mục liên quan đến Demo

```text
lung-lesion-segmentation/
├── demo/
│   ├── app.py                     # Entry point chính của ứng dụng Streamlit
│   ├── inference_engine.py        # Module nạp model, HU windowing, inference & metrics
│   └── visualization_tools.py     # Module tạo bản đồ nhiệt, overlay và plot đồ thị
├── checkpoints/                   # Chứa trọng số đã huấn luyện (.pth)
│   ├── EXP-ATTUNET-001_best.pth   # Trọng số tốt nhất của Attention U-Net
│   ├── EXP-UNET-001_best.pth      # Trọng số Standard U-Net
│   └── EXP-SEGNET-001_best.pth    # Trọng số SegNet
├── reports/
│   └── model_comparison_summary.csv # Bảng tổng hợp kết quả đánh giá
└── logs/                          # File log tiến trình train (.log / .csv)
```

---

## 3. Yêu cầu hệ thống

| Thành phần | Tối thiểu (Minimum) | Khuyến nghị (Recommended) |
|---|---|---|
| **Hệ điều hành** | Windows 10/11, macOS, hoặc Ubuntu 20.04+ | Windows 10/11 Pro 64-bit |
| **Python** | Python 3.10 hoặc 3.11 | Python 3.11 |
| **RAM** | 8 GB RAM | 16 GB RAM |
| **GPU** | Không bắt buộc (Chạy tốt trên CPU) | NVIDIA GPU (VRAM ≥ 4GB, CUDA 11.8+) |
| **Trình duyệt** | Chrome, Edge, Firefox, Brave | Google Chrome / Microsoft Edge |

---

## 4. Các bước cài đặt môi trường (Setup)

Mở terminal (trên Windows dùng **PowerShell** hoặc **Command Prompt**).

### Bước 4.1: Di chuyển vào thư mục dự án
```powershell
cd D:\lung-lesion-segmentation
```

### Bước 4.2: Khởi tạo và kích hoạt môi trường ảo (Khuyến nghị)
* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(Nếu gặp lỗi `Execution_Policies` trên PowerShell, chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi chạy lại lệnh activate).*

* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 4.3: Cài đặt các thư viện phụ thuộc
Chạy lệnh cài đặt từ file `requirements.txt`:
```powershell
pip install -r requirements.txt
```

Nếu muốn cài đặt nhanh riêng cho Web Demo:
```powershell
pip install streamlit torch torchvision numpy matplotlib pillow pandas nibabel
```

### Bước 4.4: Kiểm tra cài đặt thành công
```powershell
python -c "import streamlit, torch, matplotlib, PIL, pandas; print('>>> Môi trường sẵn sàng 100%!')"
```

---

## 5. Chuẩn bị Checkpoint mô hình

Download checkpoint tại thư mục `https://drive.google.com/drive/folders/1XLIljKO6_qkflcr180LkyfQZjVg2B2aX`  rồi đưa vào `checkpoints/` trong sourcecode local:
- `checkpoints/EXP-ATTUNET-001_best.pth`: Attention U-Net.
- `checkpoints/EXP-UNET-001_best.pth`: Standard U-Net.
- `checkpoints/EXP-SEGNET-001_best.pth`: SegNet.

> **Lưu ý:** Nếu máy thành viên mới clone dự án về mà chưa có file `.pth`:
> - Tải file `checkpoint.zip` (hoặc từ Google Drive / Kaggle output của nhóm) và giải nén vào thư mục `checkpoints/`.
> - Nếu không có checkpoint, hệ thống vẫn **hoạt động bình thường** bằng cách tự động sinh trọng số ngẫu nhiên để test toàn bộ giao diện và pipeline!

---

## 6. Hướng dẫn khởi chạy Web Demo

### Lệnh chạy chính:
Tại thư mục gốc `lung-lesion-segmentation`, thực thi lệnh:

```powershell
streamlit run demo/app.py
```

Hoặc thông qua Python module:
```powershell
python -m streamlit run demo/app.py
```

### Kết quả trên Terminal:
 #### nếu terminal báo `Email: ` bạn cần nhập email vào và sẽ tự nhảy qua web -> wait a minute
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

Trình duyệt web mặc định sẽ tự động mở trang giao diện tại địa chỉ: `http://localhost:8501`.

---

## 7. Hướng dẫn thao tác trên giao diện Web

### 7.1. Cấu hình bên thanh Sidebar (Cột bên trái)
1. **Mô hình suy luận (Inference Model):**
   - Chọn một trong ba kiến trúc: `Attention U-Net (Đề xuất)`, `Standard U-Net (Baseline)`, hoặc `SegNet (Baseline)`.
2. **Đường dẫn Checkpoint (.pth):**
   - Ứng dụng tự động phát hiện đường dẫn checkpoint mặc định trong thư mục `checkpoints/`.
   - Nếu nạp thành công, nhãn thông báo màu xanh `✓ Đã nạp checkpoint: ...` sẽ hiển thị.
3. **Phần cứng suy luận:**
   - Tự động nhận diện thiết bị `CUDA` (nếu có card đồ họa NVIDIA) hoặc `CPU`.
4. **Cửa sổ hiển thị CT (HU Windowing):**
   - Mặc định: `Cửa sổ Phổi [-1000, 400] HU` (Tối ưu nhất cho phát hiện kính mờ và đông đặc phổi).
   - Tùy chọn khác: Trung thất, Xương, hoặc Ảnh nguyên bản.
5. **Ngưỡng nhị phân (Probability Threshold):**
   - Mặc định $0.50$. Có thể kéo từ $0.1$ đến $0.9$ để tăng/giảm độ nhạy (Sensitivity vs Specificity).
6. **Độ mờ lớp phủ màu (Overlay Alpha):**
   - Điều chỉnh độ trong suốt của mặt nạ tổn thương để quan sát được cấu trúc giải phẫu bên dưới.

---

### 7.2. Tab 1: 🔬 Chẩn đoán Lát cắt Đơn (Single Slice)
Đây là màn hình làm việc chính của bác sĩ/người dùng:

1. **Chọn nguồn dữ liệu CT:**
   - **Cách 1: Ca bệnh mẫu (Synthetic Clinical Case):**
     - Hệ thống tự sinh khối CT 3D 40 lát cắt theo giải phẫu phổi thật.
     - Kéo thanh trượt `Chọn lát cắt theo trục dọc (Z-axis)` để duyệt từng lát cắt qua phổi. Hệ thống tự động chọn sẵn lát cắt có tổn thương rõ nhất.
   - **Cách 2: Tải lên file (Upload .nii / .png / .jpg):**
     - Nhấn kéo-thả file ảnh CT cắt lớp 2D (`.png`, `.jpg`) hoặc thể tích NIfTI y tế (`.nii`, `.nii.gz`).
2. **Quan sát kết quả phân đoạn:**
   - **Khung 1 (CT gốc):** Ảnh sau khi áp dụng HU Window nhu mô phổi.
   - **Khung 2 (Heatmap):** Bản đồ nhiệt phân bố xác suất tổn thương ($0.0 \rightarrow 1.0$).
   - **Khung 3 (Phân đoạn AI):** Lớp phủ màu đỏ thể hiện vùng tổn thương được AI trích xuất.
   - **Khung 4 (Ground Truth):** Nhãn tham chiếu chuyên gia y tế (nếu có).
3. **Báo cáo định lượng y khoa tự động (4 thẻ KPI phía dưới):**
   - **Diện tích tổn thương:** Tính bằng Pixel và quy đổi ra $cm^2$ thực tế.
   - **Tỷ lệ gánh nặng phổi (Burden %):** Thể hiện % tổn thương so với toàn bộ dung tích phổi.
   - **Mức độ nghiêm trọng:** Đánh giá tự động:
     - 🟢 *Nhẹ* ($< 25\%$)
     - 🟡 *Trung bình* ($25\% - 50\%$)
     - 🔴 *Nặng* ($> 50\%$)
   - **Thời gian suy luận (Latency):** Thời gian xử lý cho 1 lát cắt (thường $10 - 30\text{ ms}$ trên GPU, $\approx 50 - 150\text{ ms}$ trên CPU).

---

### 7.3. Tab 2: ⚖️ So sánh Đa mô hình (Model Comparison)
Chức năng dành cho nghiên cứu và đối chuẩn:

1. Chọn **Mô hình A** (Ví dụ: `Attention U-Net`) và **Mô hình B** (Ví dụ: `Standard U-Net`).
2. Quan sát kết quả trực quan song song:
   - Cơ chế **Attention Gate** giúp làm sạch các vùng nhiễu ngoài rìa ngực, ức chế các đốm báo động giả (False Positives) thường gặp ở U-Net truyền thống.
3. So sánh trực tiếp: Số điểm ảnh phát hiện và thời gian trễ của từng mô hình.

---

### 7.4. Tab 3: 📊 Bảng số liệu Nghiên cứu & Benchmark
1. **Bảng tổng hợp Benchmark:** Tự động nạp bảng từ `reports/model_comparison_summary.csv` với các chỉ số Dice, IoU, Precision, Recall trên tập dữ liệu kiểm thử độc lập.
2. **Đồ thị học tập (Learning Curves):** Hiển thị trực quan quá trình giảm Loss và cải thiện Dice qua các epoch từ thư mục `logs/`.

---

## 8. Các lỗi thường gặp và cách khắc phục (Troubleshooting)

### 1. Lỗi: `ModuleNotFoundError: No module named 'streamlit'` (hoặc `torch`, `nibabel`)
- **Nguyên nhân:** Chưa kích hoạt môi trường ảo hoặc chưa cài đặt thư viện.
- **Khắc phục:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

### 2. Lỗi xung đột cổng: `Port 8501 is already in use`
- **Nguyên nhân:** Đang có một tiến trình demo khác đang chạy ở cổng 8501.
- **Khắc phục:** Chạy trên một cổng khác:
  ```powershell
  streamlit run demo/app.py --server.port 8502
  ```

### 3. Lỗi PowerShell không cho chạy file script: `cannot be loaded because running scripts is disabled`
- **Khắc phục:** Mở PowerShell với tư cách User hiện tại và chạy:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

### 4. Lỗi hiển thị font chữ tiếng Việt trên terminal Windows
- **Khắc phục:** Thêm biến môi trường UTF-8 trước khi chạy:
  ```powershell
  $env:PYTHONUTF8 = "1"
  streamlit run demo/app.py
  ```

### 5. Muốn chia sẻ Web Demo qua mạng nội bộ LAN / Wi-Fi cho các thành viên cùng phòng
- **Khắc phục:** Chạy kèm cờ `--server.address 0.0.0.0`:
  ```powershell
  streamlit run demo/app.py --server.address 0.0.0.0
  ```
  Các thành viên khác chỉ cần nhập `http://<IP_MÁY_BẠN>:8501` trên trình duyệt điện thoại hoặc laptop cùng mạng Wi-Fi là truy cập được ngay.

---

## 9. Kịch bản đề xuất khi thuyết trình Demo (Presentation Flow)

Khi các thành viên dùng Web Demo để báo cáo tiến độ tuần hoặc thuyết trình đồ án:
1. **Mở đầu (1 phút):** Giới thiệu kiến trúc tổng thể trên Sidebar, chỉ ra thiết bị phần cứng (GPU/CPU) và checkpoint đã nạp.
2. **Trình diễn chẩn đoán (2 phút):**
   - Chọn ca bệnh mẫu -> Kéo thanh trượt lát cắt Z để thấy tổn thương xuất hiện dần và đậm nhất.
   - Trỏ vào bản đồ nhiệt xác suất và lớp phủ màu đỏ.
   - Đọc kết quả định lượng: Diện tích $cm^2$, gánh nặng phổi % và cảnh báo mức độ nghiêm trọng.
3. **Thuyết phục về cải tiến kỹ thuật (2 phút):**
   - Chuyển sang **Tab 2 (So sánh Đa mô hình)**.
   - Đối chiếu Attention U-Net vs Standard U-Net: chỉ rõ Standard U-Net bị báo động giả ở thành ngực và mô mềm, trong khi Attention Gate đã triệt tiêu hoàn toàn nhiễu này.
4. **Kết luận (1 phút):**
   - Chuyển sang **Tab 3** để chiếu bảng số liệu thực nghiệm định lượng và đường cong hội tụ.

