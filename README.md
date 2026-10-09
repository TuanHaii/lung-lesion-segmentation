# Lung Lesion Segmentation

<<<<<<< HEAD
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
=======
Phân đoạn tổn thương phổi từ ảnh CT sử dụng Attention U-Net, huấn luyện trên tập dữ liệu COVID-19 CT 2D slice.

---

## Yêu cầu hệ thống

- Python 3.8+
- CUDA (khuyến nghị, có thể chạy CPU)

---

## Cài đặt

```bash
git clone https://github.com/your-username/lung-lesion-segmentation.git
cd lung-lesion-segmentation

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Kiểm tra môi trường:

```bash
python test_env.py
```

---

## Cấu trúc thư mục

```
lung-lesion-segmentation/
├── configs/              # File cấu hình thí nghiệm (YAML)
├── data/
│   ├── raw/
│   │   ├── ct_volumes/           # CT volumes gốc (.nii/.nii.gz)
│   │   └── ground_truth_masks/   # Mask nhãn gốc (.nii/.nii.gz)
│   └── processed/                # Slice 2D đã xử lý + manifest CSV
├── losses/               # HybridLoss (BCE + Dice)
├── scripts/              # Script tiền xử lý dữ liệu
├── src/
│   ├── data/             # HU windowing, normalize, augmentation
│   ├── models/           # UNet, Attention UNet
│   ├── losses/           # Loss functions
│   └── utils/            # Metrics (Dice, IoU, Precision, Recall)
├── tests/                # Kiểm thử pipeline
├── train.py              # Entry point huấn luyện
└── requirements.txt
```

---

## Dữ liệu

Tải dataset từ Zenodo: [10.5281/zenodo.3757476](https://doi.org/10.5281/zenodo.3757476)

Giải nén và đặt vào:
```
data/raw/ct_volumes/          ← các file CT (.nii/.nii.gz)
data/raw/ground_truth_masks/  ← các file mask tương ứng
```

---

## Pipeline tiền xử lý

Chạy lần lượt các script sau:

**1. Kiểm tra tính toàn vẹn dữ liệu thô:**
```bash
python scripts/inspect_raw_data.py
```
Xuất báo cáo JSON tại `reports/DATA-03/Raw_Data_Integrity_Report.json`.

**2. Sinh MD5 checksum:**
```bash
python scripts/generate_checksums.py
```
Xuất `data/raw/md5_checksums.txt`.

**3. Lọc và lấy mẫu slice:**
```bash
python scripts/filter_slices.py
```
Lọc theo diện tích phổi (≥3000 px) và lesion (≥15 px), lấy mẫu đều 1.450 lesion + 300 normal slice, xuất `data/processed/clean_slices_manifest.csv`.

**4. Chia tập train/val/test theo bệnh nhân:**
```bash
python scripts/split_patient_level.py
```
Chia 70/15/15 theo bệnh nhân (seed=2026), không rò rỉ dữ liệu giữa các tập, xuất `data/processed/splits_patient_level.json`.

---

## Huấn luyện

```bash
python train.py --config configs/config.yaml
```

Cấu hình mặc định (`configs/config.yaml`):
- Model: Attention U-Net (in=1, out=1)
- Input: 256×256, batch=16
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR (T_max=100)
- Loss: BCE × 0.5 + Dice × 0.5
- Early stopping: patience=15 trên val Dice
- Checkpoint tốt nhất: `checkpoints/EXP-ATTUNET-001_best.pth`

Để chạy baseline U-Net:
```bash
python train.py --config configs/baseline_unet.yaml
```

---

## Kiểm thử

```bash
python tests/test_pipeline.py
```

Chạy forward pass → loss → backward → optimizer step → tính Dice/IoU/Precision/Recall trên dummy tensor để xác nhận pipeline hoạt động.

---

## Metrics đánh giá

| Metric | Mô tả |
|--------|-------|
| Dice (DSC) | Độ trùng khớp giữa dự đoán và ground truth |
| IoU (Jaccard) | Tỉ lệ giao/hợp |
| Precision | Tỉ lệ dự đoán đúng lesion |
| Recall | Tỉ lệ lesion thật được phát hiện |

---

## Dataset

COVID-19 CT Lung and Infection Segmentation Dataset  
Zenodo DOI: [10.5281/zenodo.3757476](https://doi.org/10.5281/zenodo.3757476) — CC BY 4.0  
20 CT volumes, nhãn: `{0: nền, 1: phổi trái, 2: phổi phải, 3: vùng nhiễm trùng}`
>>>>>>> main
