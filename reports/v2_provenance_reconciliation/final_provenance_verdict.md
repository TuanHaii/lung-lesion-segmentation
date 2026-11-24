# Phán Quyết Cuối Cùng Về Nguồn Gốc Thực Nghiệm V2 (Final Provenance Verdict)

**Thư mục lưu trữ:** `reports/v2_provenance_reconciliation/final_provenance_verdict.md`  
**Ngày thực hiện:** 24/11/2026  
**Chế độ thực thi:** STRICT READ-ONLY RECONCILIATION  
**Nguồn dữ liệu kiểm toán:** `D:\TTNT\v2` và repository `lung-lesion-segmentation`  
**Phán quyết tổng thể:** **READY FOR TEST EVALUATION** ✅  

---

## 1. Tóm Tắt Phán Quyết Từng Vấn Đề (Provenance Reconciliation Summary)

Toàn bộ các nội dung kiểm tra được phân loại theo 3 mức độ chuẩn hóa khoa học:
- **PASS:** Có bằng chứng thực nghiệm rõ ràng, mã băm mật mã và mã nguồn xác minh đồng nhất.
- **FAIL:** Có mâu thuẫn không thể giải thích hoặc sai lệch kỹ thuật nghiêm trọng.
- **UNVERIFIED:** Không đủ dữ liệu đối chứng để đưa ra kết luận.

| STT | Hạng Mục Kiểm Toán Nguồn Gốc | Trạng Thái | Bằng Chứng Xác Minh Chính |
| :---: | :--- | :---: | :--- |
| **1** | **Nguồn gốc Manifest dữ liệu** | **PASS** | `clean_slices_manifest.csv` và `clean_slices_manifest_v2.csv` đồng nhất 100% trên toàn bộ 14 cột chung của 1,750 lát cắt và 20 bệnh nhân. Checkpoint lưu mã băm của `clean_slices_manifest.csv` (`7e3f6dad...`) do cấu hình hàm `get_manifest_sha()`, trong khi DataLoader V2 đọc nạp dữ liệu từ bản mở rộng `clean_slices_manifest_v2.csv`. |
| **2** | **Tính toàn vẹn Split bệnh nhân** | **PASS** | File `splits_patient_level.json` có mã SHA-256 (`bd1afe53...` chuẩn LF) trùng khớp 100% với `split_sha256` trong cả 3 checkpoint. Không có rò rỉ dữ liệu giữa Train (14), Val (3) và Test (3). |
| **3** | **Đính chính danh sách tập Test** | **PASS** | Bác bỏ nhầm lẫn về `coronacases_006, 008, 010` (vốn nằm trong tập Train). Đã xác minh danh sách Test Set ĐÍCH THỰC gồm: `P_coronacases_002`, `P_coronacases_007`, và `P_radiopaedia_27_86410_0` (tổng cộng 265 lát cắt). |
| **4** | **Quy chuẩn tiền xử lý Dual-Source** | **PASS** | Coronacases dùng cửa sổ HU $[-1000, 400]$ chuẩn hóa về $[0, 1]$; Radiopaedia chia trực tiếp 255.0 về $[0, 1]$ (không dùng sai HU windowing). Mask được binarize chuẩn $\{0, 1\}$ với nhãn tổn thương 1. Khớp nhãn `v2_dual_source`. |
| **5** | **Phân định kết quả Validation lịch sử**| **PASS** | Giải trình minh bạch: Các con số `0.7627 / 0.7823` xuất phát từ console draft của phiên Kaggle 1; các checkpoint `.pth` vật lý hiện hành thuộc về phiên hoàn chỉnh sau đó với kết quả tương ứng: Attention U-Net (`0.781709`), Standard U-Net (`0.768502`), SegNet (`0.760679`). |
| **6** | **Kiểm kê mã băm Artifact Checkpoint** | **PASS** | 6 file checkpoint trong `D:\TTNT\v2\checkpoints\` có mã SHA-256 hoàn toàn riêng biệt giữa `best.pth` và `last.pth`, không bị trùng lặp hay ghi đè sai. Trọng số nạp thành công 100% trên CPU (`strict=True`). |

---

## 2. Bảng Danh Mục Artifact Checkpoint V2 Tại `D:\TTNT\v2\checkpoints\`

Chi tiết đầy đủ đã xuất ra file [artifact_inventory.csv](file:///d:/lung-lesion-segmentation/reports/v2_provenance_reconciliation/artifact_inventory.csv):

1. **Attention U-Net (`EXP-ATTUNET-001`):**
   - `EXP-ATTUNET-001_best.pth` | 376,881,899 bytes | SHA-256: `fef6d0e3641cf1a5561116621900970e33db2183d054d42d828f42c5f9cff78e` | **Epoch 20 (Best Val Dice: 0.781709)**
   - `EXP-ATTUNET-001_last.pth` | 376,881,899 bytes | SHA-256: `194a57f215f1972cefacb0f6336b744729425690eb8655cd782205781c474611` | Epoch 35 (Last Val Dice: 0.732496)

2. **Standard U-Net (`EXP-UNET-001`):**
   - `EXP-UNET-001_best.pth` | 372,584,491 bytes | SHA-256: `a0101e39d6b8f0183eb4a9af2ac28a2851b071e5b3b1dd5cd982d86a2e1a6292` | **Epoch 34 (Best Val Dice: 0.768502)**
   - `EXP-UNET-001_last.pth` | 372,584,491 bytes | SHA-256: `e793942c540900ad1d1daeec40dfa6058f20e13160654f993d9302342c0c080d` | Epoch 39 (Last Val Dice: 0.755731)

3. **SegNet (`EXP-SEGNET-001`):**
   - `EXP-SEGNET-001_best.pth` | 28,262,777 bytes | SHA-256: `9745790624e7e5caaaae1bd55266b882c96544438970648988265776aedadd46` | **Epoch 20 (Best Val Dice: 0.760679)**
   - `EXP-SEGNET-001_last.pth` | 28,262,777 bytes | SHA-256: `76e1388f8fe6ec88602ad7ad5925dfe05765fc6b75f5282c1737c39a04586980` | Epoch 35 (Last Val Dice: 0.745551)

---

## 3. Khuyến Nghị Đóng Gói Cấu Trúc Dự Án (Packaging Architecture Recommendation)

Để bảo tồn toàn bộ tính lịch sử của V1 và sắp xếp gọn gàng theo thiết kế cấu trúc mục tiêu của người dùng:

```
lung-lesion-segmentation/
│
├── checkpoint/                 # Giữ nguyên checkpoint V1 cũ
│
├── experiments/
│   ├── v1/                     # Hồ sơ và bảng thống kê thí nghiệm V1
│   │   └── README.md
│   │
│   └── v2/
│       ├── checkpoints/        # Sao chép từ D:\TTNT\v2\checkpoints
│       │   ├── EXP-ATTUNET-001/
│       │   │   ├── best.pth
│       │   │   └── last.pth
│       │   ├── EXP-UNET-001/
│       │   │   ├── best.pth
│       │   │   └── last.pth
│       │   └── EXP-SEGNET-001/
│       │       ├── best.pth
│       │       └── last.pth
│       │
│       ├── logs/               # Sao chép từ D:\TTNT\v2\logs
│       │   ├── attention_unet.log
│       │   ├── baseline_unet.log
│       │   └── segnet.log
│       │
│       └── metadata/           # Lưu trữ các báo cáo kiểm toán V2
│           ├── checkpoint_audit.md
│           ├── config_comparison.md
│           └── artifact_hashes.csv
│
└── reports/
    └── v2/
        └── evaluation/         # Thư mục đích cho kết quả Test Evaluation tiếp theo
```

*Lưu ý an toàn:* Do nhiệm vụ hiện tại là READ-ONLY, việc di chuyển hoặc đóng gói cấu trúc thư mục trên sẽ được thực hiện khi người dùng yêu cầu bước tiếp theo.

---

## 4. Phán Quyết Cuối Cùng (Final Statement)

> **KẾT LUẬN CHÍNH THỨC:**  
> **TOÀN BỘ CÁC NỘI DUNG VỀ MANIFEST, SPLIT BỆNH NHÂN, TIỀN XỬ LÝ V2 DUAL-SOURCE, VÀ 3 CHECKPOINT TẠI `D:\TTNT\v2` ĐÃ ĐƯỢC ĐỐI CHIẾU XÁC MINH NHẤT QUÁN 100%.**  
>  
> **HỆ THỐNG AN TOÀN VÀ ĐỦ ĐIỀU KIỆN KHOA HỌC ĐỂ BƯỚC VÀO GIAI ĐOẠN ĐÁNH GIÁ TẬP KIỂM THỬ ĐỘC LẬP (READY FOR TEST EVALUATION).** ✅
