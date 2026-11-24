# Báo Cáo So Sánh Cấu Hình Thực Nghiệm (Config Comparison)

**Thư mục lưu trữ:** `reports/checkpoint_audit_20261124/config_comparison.md`  
**Ngày kiểm toán:** 24/11/2026  
**Chế độ thực thi:** READ-ONLY CHECKPOINT AUDIT  

---

## 1. Tổng Quan Cấu Hình Thực Nghiệm

Kiểm tra đối chiếu cấu hình 3 mô hình học sâu phục vụ bài toán phân đoạn tổn thương phổi do COVID-19:
1. **Attention U-Net** (`configs/attention_unet.yaml` / `EXP-ATTUNET-001`)
2. **Standard U-Net Baseline** (`configs/baseline_unet.yaml` / `EXP-UNET-001`)
3. **SegNet Baseline** (`configs/segnet.yaml` / `EXP-SEGNET-001`)

Mục tiêu so sánh nhằm đảm bảo toàn bộ điều kiện biên, siêu tham số huấn luyện, hàm mất mát, tối ưu hóa và tiền xử lý dữ liệu được kiểm soát đồng nhất (controlled experiment), chỉ khác biệt duy nhất ở kiến trúc mạng nơ-ron.

---

## 2. Bảng So Sánh Chi Tiết Từng Tham Số (Side-by-Side Comparison)

| Tham Số / Thành Phần | Attention U-Net (`EXP-ATTUNET-001`) | Standard U-Net (`EXP-UNET-001`) | SegNet (`EXP-SEGNET-001`) | Trạng Thái Nhất Quán |
| :--- | :--- | :--- | :--- | :--- |
| **Kiến trúc mô hình (`name`)** | `AttentionUNet` | `UNet` | `SegNet` | Kiến trúc chuyên biệt |
| **Kênh đầu vào (`in_channels`)** | 1 (CT Slice Gray) | 1 (CT Slice Gray) | 1 (CT Slice Gray) | **ĐỒNG NHẤT (100%)** |
| **Kênh đầu ra (`out_channels`)** | 1 (Binary Mask) | 1 (Binary Mask) | 1 (Binary Mask) | **ĐỒNG NHẤT (100%)** |
| **Tổng số tham số (Parameters)** | **31,388,013** | **31,036,481** | **2,346,433** | Khác biệt kiến trúc |
| **Kích thước ảnh (`image_size`)** | [256, 256] | [256, 256] | [256, 256] | **ĐỒNG NHẤT (100%)** |
| **Kích thước lô (`batch_size`)** | 16 | 16 | 16 | **ĐỒNG NHẤT (100%)** |
| **Số luồng DataLoader (`num_workers`)** | 2 | 2 | 2 | **ĐỒNG NHẤT (100%)** |
| **Pin Memory (`pin_memory`)** | True | True | True | **ĐỒNG NHẤT (100%)** |
| **Nhãn tổn thương (`lesion_label`)** | 1 (COVID-19 Infection) | 1 (COVID-19 Infection) | 1 (COVID-19 Infection) | **ĐỒNG NHẤT (100%)** |
| **Hạt giống ngẫu nhiên (`seed`)** | 2026 | 2026 | 2026 | **ĐỒNG NHẤT (100%)** |
| **Số Epochs cấu hình** | 100 | 100 | 100 | **ĐỒNG NHẤT (100%)** |
| **Early Stopping Patience** | 15 epochs | 15 epochs | 15 epochs | **ĐỒNG NHẤT (100%)** |
| **Thuật toán tối ưu (`optimizer`)** | AdamW | AdamW | AdamW | **ĐỒNG NHẤT (100%)** |
| **Tốc độ học cơ sở (`lr`)** | 0.0001 (1e-4) | 0.0001 (1e-4) | 0.0001 (1e-4) | **ĐỒNG NHẤT (100%)** |
| **Suy giảm trọng số (`weight_decay`)**| 0.01 | 0.01 | 0.01 | **ĐỒNG NHẤT (100%)** |
| **Bộ điều chỉnh LR (`scheduler`)** | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | **ĐỒNG NHẤT (100%)** |
| **Chu kỳ Scheduler (`T_max`)** | 100 | 100 | 100 | **ĐỒNG NHẤT (100%)** |
| **Tốc độ học tối thiểu (`eta_min`)** | 1.0e-06 (1e-6) | 1.0e-06 (1e-6) | 1.0e-06 (1e-6) | **ĐỒNG NHẤT (100%)** |
| **Hàm mất mát (`loss`)** | HybridLoss | HybridLoss | HybridLoss | **ĐỒNG NHẤT (100%)** |
| **Trọng số BCE (`alpha_bce`)** | 0.5 | 0.5 | 0.5 | **ĐỒNG NHẤT (100%)** |
| **Trọng số Dice (`beta_dice`)** | 0.5 | 0.5 | 0.5 | **ĐỒNG NHẤT (100%)** |
| **Độ mịn hằng số (`smooth_eps`)** | 1.0e-05 (1e-5) | 1.0e-05 (1e-5) | 1.0e-05 (1e-5) | **ĐỒNG NHẤT (100%)** |
| **Pipeline tiền xử lý** | `v2_dual_source` | `v2_dual_source` | `v2_dual_source` | **ĐỒNG NHẤT (100%)** |
| **Cửa sổ HU (HU Windowing)** | [-1000, 400] HU -> [0, 1] | [-1000, 400] HU -> [0, 1] | [-1000, 400] HU -> [0, 1] | **ĐỒNG NHẤT (100%)** |
| **Git Commit cơ sở** | `34f00e06ef0a410f...` | `34f00e06ef0a410f...` | `34f00e06ef0a410f...` | **ĐỒNG NHẤT (100%)** |
| **Data Manifest SHA-256** | `7e3f6dadfef2862f...` | `7e3f6dadfef2862f...` | `7e3f6dadfef2862f...` | **ĐỒNG NHẤT (100%)** |
| **Split SHA-256** | `bd1afe5325603baa...` | `bd1afe5325603baa...` | `bd1afe5325603baa...` | **ĐỒNG NHẤT (100%)** |

---

## 3. Phân Tích Sự Khác Biệt Kiến Trúc (Architecture-Specific Settings)

Sự khác biệt giữa 3 mô hình hoàn toàn bắt nguồn từ thiết kế cấu trúc mạng:

### 3.1. Attention U-Net (`features: [64, 128, 256, 512]`)
- **Đặc trưng:** Tích hợp các khối **Attention Gate (AG)** tại mỗi tầng giải mã (decoder).
- **Cơ chế:** Tín hiệu gating $g$ từ tầng sâu hơn lọc bớt nhiễu nền và kích hoạt có trọng số các đặc trưng không gian $x$ từ skip connection trước khi ghép nối (concatenation).
- **Số tham số:** **31,388,013** (~31.39M). Độ phức tạp tăng nhẹ (+351,532 tham số, tương đương +1.13%) so với Standard U-Net do các phép chiếu $W_g, W_x, \psi$ trong Attention Gate.

### 3.2. Standard U-Net (`features: [64, 128, 256, 512]`)
- **Đặc trưng:** Cấu trúc U-Net kinh điển (Ronneberger et al.) gồm 4 tầng encoder, 1 bottleneck (1024 channels), và 4 tầng decoder.
- **Cơ chế:** Ghép nối trực tiếp (plain channel concatenation) đặc trưng không gian thô từ encoder sang decoder.
- **Số tham số:** **31,036,481** (~31.04M).

### 3.3. SegNet (`init_features: 32`)
- **Đặc trưng:** Cấu trúc đối xứng dựa trên VGG-16 với 5 tầng encoder và 5 tầng decoder. Kênh khởi tạo là 32 (32 -> 64 -> 128 -> 256 -> 256).
- **Cơ chế:** Không truyền toàn bộ feature map qua skip connection. Thay vào đó, SegNet ghi nhớ **Max-Pooling Indices** tại encoder và thực hiện **Max-Unpooling** tại decoder, giảm thiểu tối đa dung lượng bộ nhớ.
- **Số tham số:** **2,346,433** (~2.35M). Nhẹ hơn khoảng 13.3 lần so với U-Net và Attention U-Net.

---

## 4. Đối Chiếu YAML Config Gốc vs Embedded Config Trong Checkpoint

Đã kiểm tra đối chiếu trực tiếp giữa file YAML tại thư mục `configs/` và đối tượng `ckpt['config']` được lưu trong file checkpoint `.pth`:
- **Attention U-Net:** 0 khác biệt (`diffs = []`).
- **Standard U-Net:** 0 khác biệt (`diffs = []`).
- **SegNet:** 0 khác biệt (`diffs = []`).

Toàn bộ các siêu tham số huấn luyện chạy trên thực tế khớp chính xác 100% với định nghĩa cấu hình kiểm soát phiên bản.

---

## 5. Kết Luận Kiểm Toán Cấu Hình

Cả ba mô hình đều thỏa mãn nguyên tắc kiểm soát thực nghiệm y khoa nghiêm ngặt:
1. **Kiểm soát biến độc lập:** Chỉ kiến trúc mạng thay đổi.
2. **Cố định biến phụ thuộc & môi trường:** Dữ liệu, phân chia bệnh nhân, hạt giống ngẫu nhiên, tiền xử lý, hàm loss, tối ưu hóa và độ phân giải ảnh hoàn toàn đồng nhất.
3. **Tính tái lập:** Đầy đủ thông số git commit và checksum dữ liệu.
