# BÁO CÁO NGHIỆM THU KIỂM TOÁN PIPELINE V2 TRÊN KAGGLE (REAL DATA AUDIT)

**Audit Date:** 2026-11-23  
**Target Notebook:** [hainguyentuan/notebook0972614183](https://www.kaggle.com/code/hainguyentuan/notebook0972614183/edit)  
**Auditor:** Antigravity AI Agent  
**Audit Scope:** Kaggle Runtime Environment, Pre-import SHA-256 Code Gate, 20-Patient Mapping & Pairing, NIfTI Data Integrity, Full DataLoader Iteration (1.750 slices), và Tiêu chí Nghiệm thu 11 mục.

---

## 1. TỔNG KẾT NGHIỆM THU (EXECUTIVE SUMMARY)

> [!IMPORTANT]
> **KẾT LUẬN TOÀN CUỘC: 100% PASS (READY FOR TRAINING)**  
> Pipeline V2 và bộ giải quyết đường dẫn an toàn (`SafePathResolver`) đã khắc phục triệt để các biến thể tên file thực tế, giải quyết chính xác tuyệt đối **20/20 bệnh nhân** (10 Coronacases + 10 Radiopaedia) mà không có bất kỳ false-positive hay ghép nhầm cặp nào.  
> DataLoader đã nạp thành công **1.750/1.750 lát cắt** (110 batches) qua toàn bộ Train/Val/Test với tensor chuẩn hóa hợp lệ, không có ngoại lệ, không có NaN/Inf, và đạt đủ 11/11 tiêu chí nghiệm thu khắt khe.

### Điểm nổi bật sau kiểm toán:
1. **Pre-Import SHA-256 Gate:** Mã nguồn `resolver.py` và `dataset.py` trên Kaggle khớp chính xác **100% bit-for-bit** với bản cập nhật local (`4a3458ba...` và `d4fcfaf4...`).
2. **Patient Mapping:** Đạt **20/20 bệnh nhân** (khắc phục hoàn toàn lỗi Case 007, 009, 010 và 10 ca Radiopaedia). Mỗi bệnh nhân có duy nhất 1 CT và 1 Mask hợp lệ.
3. **NIfTI Integrity:** Tất cả 20 cặp đều có `diff == 0.0e+00` (ma trận affine trùng khớp hoàn hảo), orientation đồng nhất `LAS`, hình dạng không gian khớp 100%, dải nhãn chỉ thuộc `{0, 1}`, không phát hiện dữ liệu lỗi.
4. **Full DataLoader Iteration:** Duyệt qua toàn bộ **1.277 slices Train (80 batches)**, **208 slices Val (13 batches)**, **265 slices Test (17 batches)** mà không cần model forward hay gradient.

---

## 2. BẢNG ĐỐI CHIẾU 11 TIÊU CHÍ NGHIỆM THU (PASS/FAIL)

| STT | Tiêu chí | Điều kiện PASS | Kết quả thực tế trên Kaggle | Đánh giá |
|:---:|---|---|---|:---:|
| 1 | **Code fingerprint** | 2/2 file khớp SHA-256 | `resolver.py` và `dataset.py` khớp 100% SHA-256 local | **PASS** |
| 2 | **Patient mapping** | 20/20 bệnh nhân | 20/20 bệnh nhân có đúng 1 CT & 1 Mask duy nhất | **PASS** |
| 3 | **CT/Mask pairing** | Không sai source, case, study, series | Tất cả 20 cặp đồng nhất canonical ID và source | **PASS** |
| 4 | **NIfTI integrity** | Shape khớp, affine diff trong tolerance (1e-3) | 20/20 shape khớp, diff = 0.0e+00, orientation LAS khớp | **PASS** |
| 5 | **Mask integrity** | Nhãn hợp lệ, foreground hợp lý | Labels chỉ thuộc {0, 1}, foreground 0.0% - 4.17% | **PASS** |
| 6 | **DataLoader Train** | Đủ 1.277 slices | 1.277/1.277 slices (80 batches, không crash) | **PASS** |
| 7 | **DataLoader Val** | Đủ 208 slices | 208/208 slices (13 batches, không crash) | **PASS** |
| 8 | **DataLoader Test** | Đủ 265 slices | 265/265 slices (17 batches, không crash) | **PASS** |
| 9 | **Source coverage** | Có cả Coronacases và Radiopaedia | Mọi split đều chứa cả Coronacases và Radiopaedia | **PASS** |
| 10 | **Label coverage** | Có cả positive và negative slices | Mọi split đều có positive (~83%) và negative (~17%) | **PASS** |
| 11 | **Runtime** | Không có exception, NaN hoặc Inf | 1.750 slices hữu hạn (`finite=True`), 0 exception | **PASS** |

---

## 3. THÔNG SỐ MÔI TRƯỜNG THỰC TẾ TRÊN KAGGLE

- **Python Version:** `3.13.16`
- **PyTorch Version:** `2.11.0+cu130` (CUDA Available: `True`, GPU: `NVIDIA Tesla T4`)
- **NumPy Version:** `2.1.3`
- **Nibabel Version:** `5.4.2`
- **Active Working Directory:** `/kaggle/working/lung-lesion-segmentation`
- **Dataset Mount Path:** `/kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset`
- **Search Roots Đã Quét:**
  - `/kaggle/input`
  - `/kaggle/input/datasets`
  - `/kaggle/input/datasets/hainguyentuan`
  - `/kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset`
  - `/kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases`
  - `/kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask`

---

## 4. CHI TIẾT MAPPING & TOÀN VẸN DỮ LIỆU NIFTI 20 BỆNH NHÂN

| Canonical Patient ID | Nguồn | Kích thước (Shape) | Hướng | Sai khác Affine | Nhãn Mask | Voxel tổn thương (%) | Dải giá trị CT | Trạng thái |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `P_coronacases_001` | coronacases | (512, 512, 301) | LAS | 0.0e+00 | [0, 1] | 652,289 (0.83 %) | [-1021.0, 2996.0] | **PASS** |
| `P_coronacases_002` | coronacases | (512, 512, 200) | LAS | 0.0e+00 | [0, 1] | 273,835 (0.52 %) | [-1023.0, 9567.0] | **PASS** |
| `P_coronacases_003` | coronacases | (512, 512, 200) | LAS | 0.0e+00 | [0, 1] | 1,274,604 (2.43 %) | [-1023.0, 8931.0] | **PASS** |
| `P_coronacases_004` | coronacases | (512, 512, 270) | LAS | 0.0e+00 | [0, 1] | 136,376 (0.19 %) | [-1021.0, 2020.0] | **PASS** |
| `P_coronacases_005` | coronacases | (512, 512, 290) | LAS | 0.0e+00 | [0, 1] | 176,347 (0.23 %) | [-1021.0, 5528.0] | **PASS** |
| `P_coronacases_006` | coronacases | (512, 512, 213) | LAS | 0.0e+00 | [0, 1] | 149,602 (0.27 %) | [-1023.0, 2217.0] | **PASS** |
| `P_coronacases_007` | coronacases | (512, 512, 249) | LAS | 0.0e+00 | [0, 1] | 180,552 (0.28 %) | [-1023.0, 2515.0] | **PASS** |
| `P_coronacases_008` | coronacases | (512, 512, 301) | LAS | 0.0e+00 | [0, 1] | 552,371 (0.70 %) | [-1021.0, 8575.0] | **PASS** |
| `P_coronacases_009` | coronacases | (512, 512, 256) | LAS | 0.0e+00 | [0, 1] | 224,197 (0.33 %) | [-1021.0, 1845.0] | **PASS** |
| `P_coronacases_010` | coronacases | (512, 512, 301) | LAS | 0.0e+00 | [0, 1] | 864,062 (1.10 %) | [-1021.0, 1920.0] | **PASS** |
| `P_radiopaedia_10_85902_1` | radiopaedia | (630, 630, 39) | LAS | 0.0e+00 | [0, 1] | 57,373 (0.37 %) | [0.0, 255.0] | **PASS** |
| `P_radiopaedia_10_85902_3` | radiopaedia | (630, 630, 418) | LAS | 0.0e+00 | [0, 1] | 567,710 (0.34 %) | [0.0, 255.0] | **PASS** |
| `P_radiopaedia_14_85914_0` | radiopaedia | (630, 401, 110) | LAS | 0.0e+00 | [0, 1] | 1,129,152 (4.06 %) | [0.0, 255.0] | **PASS** |
| `P_radiopaedia_27_86410_0` | radiopaedia | (630, 630, 66) | LAS | 0.0e+00 | [0, 1] | 291,467 (1.11 %) | [4.0, 255.0] | **PASS** |
| `P_radiopaedia_29_86490_1` | radiopaedia | (630, 630, 42) | LAS | 0.0e+00 | [0, 1] | 163 (0.00 %) | [0.0, 255.0] | **PASS** |
| `P_radiopaedia_29_86491_1` | radiopaedia | (630, 630, 42) | LAS | 0.0e+00 | [0, 1] | 18,062 (0.11 %) | [0.0, 255.0] | **PASS** |
| `P_radiopaedia_36_86526_0` | radiopaedia | (630, 630, 45) | LAS | 0.0e+00 | [0, 1] | 66,804 (0.37 %) | [0.0, 255.0] | **PASS** |
| `P_radiopaedia_40_86625_0` | radiopaedia | (630, 630, 93) | LAS | 0.0e+00 | [0, 1] | 1,538,940 (4.17 %) | [12.0, 255.0] | **PASS** |
| `P_radiopaedia_4_85506_1` | radiopaedia | (630, 630, 39) | LAS | 0.0e+00 | [0, 1] | 415,201 (2.68 %) | [0.0, 255.0] | **PASS** |
| `P_radiopaedia_7_85703_0` | radiopaedia | (630, 630, 45) | LAS | 0.0e+00 | [0, 1] | 382,406 (2.14 %) | [0.0, 255.0] | **PASS** |

---

## 5. DANH SÁCH 20 ĐƯỜNG DẪN CT & MASK THỰC TẾ TRÊN KAGGLE

```
01. P_coronacases_001 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_001.nii/coronacases_org_001.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_001.nii
02. P_coronacases_002 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_002.nii/coronacases_org_002.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_002.nii
03. P_coronacases_003 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_003.nii/coronacases_org_003.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_003.nii
04. P_coronacases_004 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_004.nii/coronacases_org_004.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_004.nii
05. P_coronacases_005 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_005.nii/coronacases_org_005.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_005.nii
06. P_coronacases_006 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_006.nii/coronacases_org_006.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_006.nii
07. P_coronacases_007 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_007.nii/coronacases_org_case007.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_007.nii
08. P_coronacases_008 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_008.nii/coronacases_org_008.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_008.nii
09. P_coronacases_009 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_009.nii/case009.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_009.nii
10. P_coronacases_010 (coronacases):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/coronacases_010.nii/coronacases_org_case010.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/coronacases_010.nii
11. P_radiopaedia_10_85902_1 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_10_85902_1.nii/radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_10_85902_1.nii
12. P_radiopaedia_10_85902_3 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_10_85902_3.nii/radiopaedia_org_covid-19-pneumonia-10_85902_3-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_10_85902_3.nii
13. P_radiopaedia_14_85914_0 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_14_85914_0.nii/radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_14_85914_0.nii
14. P_radiopaedia_27_86410_0 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_27_86410_0.nii/radiopaedia_org_covid-19-pneumonia-27_86410_0-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_27_86410_0.nii
15. P_radiopaedia_29_86490_1 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_29_86490_1.nii/radiopaedia_org_covid-19-pneumonia-29_86490_1-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_29_86490_1.nii
16. P_radiopaedia_29_86491_1 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_29_86491_1.nii/radiopaedia_org_covid-19-pneumonia-29_86491_1-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_29_86491_1.nii
17. P_radiopaedia_36_86526_0 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_36_86526_0.nii/radiopaedia_org_covid-19-pneumonia-36_86526_0-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_36_86526_0.nii
18. P_radiopaedia_40_86625_0 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_40_86625_0.nii/radiopaedia_org_covid-19-pneumonia-40_86625_0-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_40_86625_0.nii
19. P_radiopaedia_4_85506_1 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_4_85506_1.nii/radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_4_85506_1.nii
20. P_radiopaedia_7_85703_0 (radiopaedia):
    CT  : /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/COVID-19-CT-Seg_20cases/radiopaedia_7_85703_0.nii/radiopaedia_org_covid-19-pneumonia-7_85703_0-dcm.nii
    Mask: /kaggle/input/datasets/hainguyentuan/lung-lesion-ct-dataset/Infection_Mask/radiopaedia_7_85703_0.nii
```

---

## 6. THỐNG KÊ DUYỆT TRỌN VẸN 100% DATALOADER

### 6.1. Tập huấn luyện (Train Split)
- **Số lượng slices:** **1.277 / 1.277 slices (80 batches, batch_size=16)**
- **Positive Slices (Có tổn thương):** `1.057` slices (82,77 %)
- **Negative Slices (Không tổn thương):** `220` slices (17,23 %)
- **Tỷ lệ nguồn:** Coronacases: `959` slices | Radiopaedia: `318` slices
- **Độ phủ bệnh nhân (14 ca):** `P_coronacases_001`, `003`, `004`, `006`, `008`, `009`, `010`, `P_radiopaedia_10_85902_1`, `10_85902_3`, `14_85914_0`, `29_86490_1`, `29_86491_1`, `4_85506_1`, `7_85703_0`.

### 6.2. Tập kiểm định (Validation Split)
- **Số lượng slices:** **208 / 208 slices (13 batches, batch_size=16)**
- **Positive Slices (Có tổn thương):** `172` slices (82,69 %)
- **Negative Slices (Không tổn thương):** `36` slices (17,31 %)
- **Tỷ lệ nguồn:** Coronacases: `125` slices | Radiopaedia: `83` slices
- **Độ phủ bệnh nhân (3 ca):** `P_coronacases_005`, `P_radiopaedia_36_86526_0`, `P_radiopaedia_40_86625_0`.

### 6.3. Tập kiểm thử (Test Split)
- **Số lượng slices:** **265 / 265 slices (17 batches, batch_size=16)**
- **Positive Slices (Có tổn thương):** `221` slices (83,40 %)
- **Negative Slices (Không tổn thương):** `44` slices (16,60 %)
- **Tỷ lệ nguồn:** Coronacases: `228` slices | Radiopaedia: `37` slices
- **Độ phủ bệnh nhân (3 ca):** `P_coronacases_002`, `P_coronacases_007`, `P_radiopaedia_27_86410_0`.

---

## 7. BẢNG ĐỐI CHIẾU MÃ HASH SHA-256 (CODE INTEGRITY)

| File | SHA-256 Local | SHA-256 Kaggle (Sau đồng bộ) | Trạng thái |
|---|---|---|:---:|
| `src/data/v2/resolver.py` | `4a3458ba46edd2026a2986a6d341b329f7b8fcc67654880ce3152ef277ab927c` | `4a3458ba46edd2026a2986a6d341b329f7b8fcc67654880ce3152ef277ab927c` | **MATCH (100%)** |
| `src/data/v2/dataset.py` | `d4fcfaf4b5e46d279bedbb451ed340807c1efe0e382d5ed534950316c38cf931` | `d4fcfaf4b5e46d279bedbb451ed340807c1efe0e382d5ed534950316c38cf931` | **MATCH (100%)** |

---

## 8. KẾT LUẬN & KIẾN NGHỊ BƯỚC TIẾP THEO

1. **Pipeline V2 hoàn toàn sẵn sàng cho Huấn luyện:** Quá trình nạp dữ liệu thật trên Kaggle không còn bất kỳ nút thắt hay lỗi kỹ thuật nào.
2. **Khuyến nghị Commit Git:** Hai file code `src/data/v2/resolver.py` và `src/data/v2/dataset.py` cùng unit test `tests/test_safe_resolver_v2.py` đã được kiểm chứng tuyệt đối trên cả local (42/42 tests pass) và Kaggle runtime (100% tiêu chí pass). Bạn có thể tiến hành `git commit` và `git push` lên GitHub để đồng bộ vĩnh viễn vào nhánh `main`.
3. **Bắt đầu Thực nghiệm Huấn luyện:** Bạn có thể tiến hành bước tiếp theo là khởi chạy training pipeline với các kiến trúc mô hình mong muốn (Attention U-Net, U-Net baseline, SegNet).
