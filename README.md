# Lung Lesion Segmentation

Demo phân đoạn tổn thương phổi trên ảnh CT lát cắt bằng **SegNet/PyTorch**.
Project hiện dùng dữ liệu CT và mask giả lập để kiểm tra preprocessing,
inference và visualization; chưa huấn luyện trên dữ liệu lâm sàng.

## Yêu cầu

- Windows 10/11 và Python 3.10+.
- PyTorch, NumPy, Matplotlib.
- `pytest` nếu muốn chạy test bằng pytest.
- `opencv-python` là tùy chọn để vẽ contour.

Không cần dataset, checkpoint hoặc GPU cho demo. Mô hình dùng trọng số ngẫu
nhiên nên kết quả chỉ phục vụ kiểm thử pipeline, không dùng chẩn đoán.

## Cài đặt

Mở PowerShell tại thư mục project:

```powershell
cd D:\Code\lung-lesion-segmentation
.\venv\Scripts\Activate.ps1
```

Nếu chưa có môi trường `venv`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy matplotlib pytest
```

Tùy chọn:

```powershell
pip install opencv-python
```

## Cách chạy

### Demo SegNet inference

Nếu PowerShell báo lỗi mã hóa tiếng Việt, chạy trước:

```powershell
$env:PYTHONUTF8 = "1"
```

Chạy demo:

```powershell
python .\demo_segnet_inference.py
```

Demo tạo volume CT giả lập `256 x 256 x 40`, chọn lát cắt có tổn thương,
chuẩn hóa theo lung window `[-1000, 400] HU`, chạy SegNet và lưu kết quả tại:

```text
evidence/demo_segnet_prediction.png
```

### Trực quan hóa CT và mask

```powershell
python .\src\utils\visualization.py
python .\src\utils\visualization.py --grid
```

### Chạy test

```powershell
python -m pytest .\tests
```

Hoặc chạy trực tiếp, không cần `pytest`:

```powershell
python .\tests\test_segnet.py
```

Test kiểm tra model nhận input `(2, 1, 256, 256)` và trả output cùng kích
thước không gian.

## Lưu ý

Muốn sử dụng thực tế cần thay dữ liệu giả lập bằng dataset CT đã chuẩn hóa,
thêm checkpoint đã huấn luyện và đánh giá bằng Dice/IoU. Kết quả hiện tại
không có giá trị chẩn đoán y khoa.
