# Tuyên Bố Sẵn Sàng Đánh Giá Tập Test (Evaluation Readiness Statement)

**Thư mục lưu trữ:** `reports/checkpoint_audit_20261124/evaluation_readiness.md`  
**Ngày lập báo cáo:** 24/11/2026  
**Chế độ thực thi:** STRICT READ-ONLY CHECKPOINT AUDIT  

---

## 1. Tuyên Bố Sẵn Sàng (Readiness Declaration)

> **KẾT LUẬN CHÍNH THỨC:**  
> **CẢ 3 CHECKPOINT ĐÃ HOÀN TOÀN ĐỦ ĐIỀU KIỆN VÀ SẴN SÀNG ĐỂ ĐƯỢC ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (INDEPENDENT TEST SET).**  
>  
> Trạng thái: **READY FOR TEST INFERENCE / EVALUATION** ✅

---

## 2. Bảng Danh Sách Tiền Điều Kiện Đã Được Thẩm Định (Precondition Checklist)

| Tiền Điều Kiện Bắt Buộc | Mô Tả & Bằng Chứng Thẩm Định | Trạng Thái |
| :--- | :--- | :--- |
| **1. Tính toàn vẹn trọng số** | Đã load thành công trên CPU với `strict=True` cho cả 3 kiến trúc (`AttentionUNet`, `UNet`, `SegNet`). 0 missing keys, 0 unexpected keys. | **ĐẠT (PASS)** |
| **2. Tập Test Cố Định Cấp Bệnh Nhân** | Cả 3 checkpoint dùng chung phân chia `splits_patient_level.json` (`split_sha256: bd1afe5325603baa...`). Tập Test gồm 3 bệnh nhân độc lập hoàn toàn (`coronacases_006`, `coronacases_008`, `coronacases_010`). | **ĐẠT (PASS)** |
| **3. Không rò rỉ dữ liệu (No Data Leakage)** | Bệnh nhân trong tập Test chưa từng xuất hiện trong tập Train hay tập Validation của bất kỳ mô hình nào. | **ĐẠT (PASS)** |
| **4. Tiền xử lý dữ liệu đồng nhất** | Cả 3 mô hình đều được huấn luyện qua pipeline `v2_dual_source`: chuẩn hóa cửa sổ HU [-1000, 400] -> [0, 1], nhãn tổn thương nhị phân {0, 1}, kích thước ảnh 256x256. | **ĐẠT (PASS)** |
| **5. Điều kiện thực nghiệm kiểm soát** | Hạt giống ngẫu nhiên 2026, Optimizer AdamW (lr=1e-4, wd=0.01), CosineAnnealingLR (T_max=100), HybridLoss (alpha=0.5, beta=0.5) đồng nhất tuyệt đối. | **ĐẠT (PASS)** |
| **6. Khớp mã nguồn repository** | Git commit trong checkpoint (`34f00e0`) trùng khớp 100% với commit HEAD hiện tại của repository. | **ĐẠT (PASS)** |
| **7. Quy chuẩn suy luận (Inference Protocol)** | Thống nhất áp dụng hàm kích hoạt `torch.sigmoid()` và ngưỡng nhị phân $\tau = 0.5$ trên cả 3 mô hình, tính toán metrics lát cắt (slice-level) và tích lũy thể tích bệnh nhân (patient-level). | **ĐẠT (PASS)** |

---

## 3. Các Checkpoint Sẽ Đưa Vào Đánh Giá

1. **Attention U-Net (`EXP-ATTUNET-001`):**
   - File: `d:\TTNT\EXP-ATTUNET-001_best.pth`
   - Epoch tối ưu: **Epoch 20**
   - Validation Dice: **0.781709**
   - Tham số: 31,388,013

2. **Standard U-Net Baseline (`EXP-UNET-001`):**
   - File: `d:\TTNT\EXP-UNET-001_best.pth`
   - Epoch tối ưu: **Epoch 34**
   - Validation Dice: **0.768502**
   - Tham số: 31,036,481

3. **SegNet Baseline (`EXP-SEGNET-001`):**
   - File: `d:\TTNT\checkpoints\EXP-SEGNET-001_best.pth`
   - Epoch tối ưu: **Epoch 20**
   - Validation Dice: **0.760679**
   - Tham số: 2,346,433

---

## 4. Hướng Dẫn Thực Hiện Đánh Giá Độc Lập Tiếp Theo (Next Steps)

Khi chuyển sang giai đoạn **Test Evaluation / Inference**, quy trình thực thi được khuyến nghị như sau:

```bash
# 1. Chạy đánh giá Test Set cho Attention U-Net
python src/evaluate.py --checkpoint d:/TTNT/EXP-ATTUNET-001_best.pth --model AttentionUNet --output_dir reports/attunet_test_v2

# 2. Chạy đánh giá Test Set cho Standard U-Net Baseline
python src/evaluate.py --checkpoint d:/TTNT/EXP-UNET-001_best.pth --model UNet --output_dir reports/unet_test_v2

# 3. Chạy đánh giá Test Set cho SegNet Baseline
python src/evaluate.py --checkpoint d:/TTNT/checkpoints/EXP-SEGNET-001_best.pth --model SegNet --output_dir reports/segnet_test_v2

# 4. Tổng hợp bảng so sánh đối đầu và kiểm định ý nghĩa thống kê (Wilcoxon Signed-Rank Test)
python scripts/compare_models.py
```

---

## 5. Kết Luận An Toàn

Không có bất kỳ rủi ro kỹ thuật, sai lệch siêu dữ liệu, hay nguy cơ thiên lệch thực nghiệm nào được ghi nhận. Việc chuyển sang bước **Test Inference** là an toàn 100%.
