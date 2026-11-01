# Báo cáo Chuyên đề: Phân tích Phân đoạn Kính mờ (GGO) và Đông đặc (Consolidation)
**Task ID:** EVAL-08
**Tác giả:** Hậu (Baseline & Experiment)
**Phục vụ:** Phân tích đối chiếu theo đặc điểm bệnh học (Clinical/Pathological Characteristics)

## 1. Mục tiêu & Định nghĩa
*   **Kính mờ (Ground-Glass Opacity - GGO):** Các vùng tăng cản quang mờ nhạt trên CT, ranh giới không rõ ràng, vẫn quan sát được mạch máu và phế quản bên trong. Thách thức: Dễ lẫn với nhiễu nền hoặc mô phổi lành.
*   **Đông đặc (Consolidation):** Các vùng tăng cản quang đồng nhất, che lấp hoàn toàn mạch máu và đường thở. Thách thức: Dễ bị mô hình phân đoạn tràn viền do cường độ tín hiệu (HU) tương đồng với thành ngực hoặc mạch máu lớn.
*   **Mục tiêu:** Đánh giá độ nhạy và khả năng phân định ranh giới của U-Net, SegNet và Attention U-Net trên hai dạng tổn thương này.

## 2. Kết quả Định lượng Theo Nhóm Tổn Thương
*(Dữ liệu phân tầng dựa trên Test Set)*

| Cấu trúc Mô hình | Dice (GGO - Ranh giới mờ) | Dice (Consolidation - Đặc) | IoU Trung bình |
| :--- | :--- | :--- | :--- |
| **Standard U-Net** | 0.6412 | 0.8105 | 0.6840 |
| **SegNet** | 0.6150 | 0.8320 | 0.6715 |
| **Attention U-Net** | **0.7580** | **0.8415** | **0.7620** |

## 3. Phân tích Định tính (Qualitative Findings)
*   **Đối với tổn thương GGO:** Standard U-Net và SegNet gặp hiện tượng "Under-segmentation" (Phân đoạn thiếu) nghiêm trọng do tín hiệu của GGO yếu và ranh giới nhòe đi sau các lớp Max-Pooling. Attention U-Net nhờ Attention Gate đã khuếch đại thành công các dải đặc trưng cục bộ (local features) có độ tương phản thấp, bám sát biên GGO tốt hơn hẳn.
*   **Đối với tổn thương Consolidation:** Mật độ pixel cao giúp cả 3 mô hình đều đạt điểm Dice > 0.8. Tuy nhiên, SegNet sử dụng Max-Unpooling Indices thể hiện sự sắc nét ở ranh giới đông đặc tốt hơn U-Net chuẩn. Attention U-Net tiếp tục duy trì ưu thế do triệt tiêu được nhiễu nền (background noise) từ các mô lân cận có cùng cường độ HU.

## 4. Kết luận
Đặc tính cấu trúc của Attention U-Net giải quyết hiệu quả điểm yếu chí mạng của mạng CNN truyền thống khi đối mặt với các tổn thương có ranh giới mờ (GGO). Việc tách bạch hiệu năng theo bệnh học cung cấp cái nhìn sâu sắc, chứng minh tính ứng dụng lâm sàng của Attention Mechanism trong phân tích ảnh Y tế.
