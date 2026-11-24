# Hồ Sơ Nguồn Gốc Mã Nguồn Git (Git Provenance Record)

**Đường dẫn lưu trữ:** `experiments/v2/metadata/git_provenance.md`  
**Ngày lập hồ sơ:** 24/11/2026  
**Chế độ thực thi:** STRICT READ-ONLY RECONCILIATION  
**Trạng thái kiểm tra Git:** **PASS (ĐỒNG NHẤT 100%)** ✅  

---

## 1. Mục Tiêu Kiểm Tra

Xác định chính xác phiên bản mã nguồn (Git commit) được sử dụng để huấn luyện cả 3 mô hình Version 2:
1. Trích xuất mã commit được ghi nhận bên trong từng file checkpoint `.pth`.
2. Truy vết mã băm đầy đủ (Full 40-character SHA-1) trong lịch sử Git của repository.
3. Xác minh tính đồng nhất về phiên bản mã nguồn giữa 3 checkpoint.
4. Đối chiếu commit của checkpoint với commit hiện tại trên nhánh làm việc (`HEAD`).

---

## 2. Bảng Đối Chiếu Git Commit Giữa Các Checkpoint

| Checkpoint | Experiment ID | Lớp Mô Hình | Git Commit Ghi Trong Checkpoint | Full SHA-1 Commit | Trạng Thái Khớp |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `EXP-ATTUNET-001_best.pth` | `EXP-ATTUNET-001` | `AttentionUNet` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | **PASS** |
| `EXP-ATTUNET-001_last.pth` | `EXP-ATTUNET-001` | `AttentionUNet` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | **PASS** |
| `EXP-UNET-001_best.pth` | `EXP-UNET-001` | `UNet` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | **PASS** |
| `EXP-UNET-001_last.pth` | `EXP-UNET-001` | `UNet` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | **PASS** |
| `EXP-SEGNET-001_best.pth` | `EXP-SEGNET-001` | `SegNet` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | **PASS** |
| `EXP-SEGNET-001_last.pth` | `EXP-SEGNET-001` | `SegNet` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | `34f00e06ef0a410f45f3e254da0e7facd997765e` | **PASS** |

---

## 3. Chi Tiết Commit Trong Lịch Sử Repository

Truy vết bằng lệnh `git show 34f00e06ef0a410f45f3e254da0e7facd997765e`:
- **Full Commit SHA:** `34f00e06ef0a410f45f3e254da0e7facd997765e`
- **Short Commit SHA:** `34f00e0`
- **Tác giả (Author):** `napole-dong <phuongdong1741@gmail.com>`
- **Thời gian (Commit Date):** `2026-11-23 18:23:40 +0700`
- **Tiêu đề Commit (Subject):** `fix(data-v2): harden SafePathResolver for Kaggle real data and verify 20/20 patient mapping`
- **Nội dung thay đổi chính:**
  - Hoàn thiện bộ giải quyết đường dẫn an toàn `SafePathResolver` để ánh xạ chính xác 20/20 bệnh nhân thực tế trên Kaggle (`/kaggle/input/covid-19-ct-scans`).
  - Tách biệt kiểm tra kiểu dữ liệu cho volume ảnh CT và ground truth mask.
  - Tích hợp pipeline tiền xử lý `v2_dual_source`.

---

## 4. Đối Chiếu Commit Của Checkpoint Với Trạng Thái Repo Hiện Tại

- **Commit của Checkpoint:** `34f00e06ef0a410f45f3e254da0e7facd997765e`
- **Commit HEAD cục bộ (`git rev-parse HEAD`):** `34f00e06ef0a410f45f3e254da0e7facd997765e`
- **Mức độ chênh lệch (Commit Divergence):** **0 commit (Khớp tuyệt đối)**.

---

## 5. Kết Luận Kiểm Toán Git

1. **Cùng phiên bản mã nguồn:** Cả ba mô hình (Attention U-Net, Standard U-Net, SegNet) đều được huấn luyện trên cùng một phiên bản mã nguồn `34f00e0`.
2. **Đồng bộ mã nguồn:** Mã nguồn trên branch hiện tại trùng khớp 100% với commit tại thời điểm huấn luyện, không có bất kỳ xung đột hay sửa đổi nào chưa commit (uncommitted modifications).
3. **Tuân thủ quy tắc:** Không thực hiện chỉnh sửa lịch sử Git, không tạo commit mới và không push remote.

**KẾT LUẬN CUỐI CÙNG: PASS ✅**
