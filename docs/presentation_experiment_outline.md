# Dàn ý Slide Thuyết trình: Phần Thực nghiệm & Đánh giá (FINAL-02)
**Người phụ trách:** Hậu (Bàn giao cho Duy lắp slide)
**Tài nguyên hình ảnh:** Trích xuất từ thư mục `paper_figures/`

---

## Slide 1: Thiết lập Thực nghiệm (Experimental Setup)
*   **Tiêu đề Slide:** Điều kiện Thực nghiệm & Tiền xử lý Dữ liệu
*   **Hình ảnh đính kèm:** `paper_figures/Fig4_CT_Preprocessing_Windowing.png`
*   **Luận điểm thuyết trình (Speaker Notes):**
    *   Toàn bộ ảnh CT được chuẩn hóa qua cửa sổ Hounsfield Nhu mô phổi (Lung Window: -1000 đến 400 HU) để làm nổi bật tổn thương.
    *   Dữ liệu được chia theo mức độ bệnh nhân (Patient-level Split) để tránh rò rỉ dữ liệu (Data Leakage) giữa tập Train/Val/Test.
    *   Hàm mất mát tối ưu: Lai ghép (Hybrid) giữa Binary Cross Entropy (BCE) và Dice Loss giúp mô hình cân bằng giữa phân loại pixel và tối ưu hình học.

## Slide 2: Động học Học tập (Learning Curves)
*   **Tiêu đề Slide:** Quá trình Hội tụ & So sánh Động học Học tập
*   **Hình ảnh đính kèm:** `paper_figures/Fig1_Learning_Curves.png`
*   **Luận điểm thuyết trình (Speaker Notes):**
    *   Cả 3 mô hình (U-Net, SegNet, Attention U-Net) đều hội tụ ổn định sau 50-100 epochs, không có hiện tượng quá khớp (overfitting) nghiêm trọng.
    *   Attention U-Net cho thấy tốc độ giảm Loss nhanh hơn và đạt Validation Dice Score cao nhất, minh chứng cho việc cơ chế Attention giúp mạng học các đặc trưng tối ưu hơn.

## Slide 3: Trực quan hóa Phân đoạn (Qualitative Visualization)
*   **Tiêu đề Slide:** Đánh giá Định tính Định vị Tổn thương
*   **Hình ảnh đính kèm:** `paper_figures/Fig2_Qualitative_Comparison.png`
*   **Luận điểm thuyết trình (Speaker Notes):**
    *   Đối chiếu 4 cột: Ảnh gốc -> Ground Truth -> Dự đoán -> Lớp phủ (Overlay).
    *   U-Net tiêu chuẩn thỉnh thoảng xuất hiện các vùng False Positive (tràn viền).
    *   Attention U-Net có khả năng khóa mục tiêu (focus) cực tốt vào vùng tổn thương, loại bỏ hiệu quả nhiễu nền từ các mô lân cận.

## Slide 4: Trả lời Câu hỏi Nghiên cứu (RQ1 & RQ2)
*   **Tiêu đề Slide:** Phân tích Chuyên sâu Tổn thương Nhỏ & Ranh giới (GGO)
*   **Hình ảnh đính kèm:** `paper_figures/Fig3_RQ2_Boundary_Analysis.png`
*   **Luận điểm thuyết trình (Speaker Notes):**
    *   **Về RQ2 (Bảo toàn ranh giới):** SegNet sử dụng Max-Unpooling Indices cho đường viền tổn thương sắc nét hơn U-Net chuẩn khi đối mặt với các khối đông đặc (Consolidation).
    *   **Về RQ1 (Kính mờ GGO):** Attention U-Net vượt trội hoàn toàn khi xử lý các tổn thương kính mờ (độ tương phản thấp). Cơ chế Attention Gate khuếch đại thành công các tín hiệu mờ nhạt, khắc phục điểm yếu bỏ sót (False Negative) của kiến trúc CNN truyền thống.
