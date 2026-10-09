# Lung Lesion Segmentation

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
