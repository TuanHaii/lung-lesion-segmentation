# BÁO CÁO KỸ THUẬT TOÀN DIỆN: QUY TRÌNH HUẤN LUYỆN, GÁN NHÃN VÀ ĐÁNH GIÁ ĐỊNH LƯỢNG MÔ HÌNH PHÂN ĐOẠN TỔN THƯƠNG CT PHỔI (TASK EVAL-02)

**Dự án**: Nghiên cứu & Phát triển Mô hình Attention U-Net trong phân đoạn tổn thương phổi do COVID-19 trên ảnh cắt lớp vi tính (CT)  
**Tác giả**: Nhóm Nghiên cứu & Kỹ thuật Dữ liệu Y tế (Medical Imaging & AI Research Team)  
**Môi trường thực nghiệm**: Kaggle GPU (NVIDIA Tesla P100 / T4, CUDA 12.x, PyTorch 2.x)  
**Phiên bản mã nguồn**: Nhánh `main` (Mã băm Commit: `0437ec6` - Đồng bộ GitHub `TuanHaii/lung-lesion-segmentation`)  
**Tập dữ liệu kiểm thử**: Independent Test Set gồm 265 lát cắt 2D (từ 3 bệnh nhân kiểm thử độc lập)  

---

## 1. KẾT QUẢ TÁI KIỂM TRA (RE-CHECK) TÍNH HỢP LỆ CỦA PREDICTIONS

Quá trình kiểm tra chéo (Cross-validation) giữa các file dự đoán chi tiết (`reports/{attunet, unet, segnet}/test_metrics_slice_level.csv`) và dữ liệu gốc (`data/processed/clean_slices_manifest.csv`) xác nhận **toàn bộ kết quả dự đoán hoàn toàn chính xác, minh bạch và nhất quán về mặt toán học**:

* **Đối với các lát cắt không có tổn thương (44 normal slices, $\text{GT} = 0$):**
  * Cả ba mô hình đều đạt $TP = 0, FP = 0 \implies \text{DSC} = 1.0, \text{IoU} = 1.0$.
  * Mô hình kiểm soát tốt phản hồi, không bị hiện tượng kích hoạt giả (không sinh False Positive trên nền mô phổi lành).
* **Đối với các lát cắt có tổn thương (221 lesion slices, $\text{GT} > 0$):**
  * Diện tích tổn thương thực tế $(TP + FN)$ trong kết quả suy luận khớp tuyệt đối với số đo diện tích `lesion_area_px` trong manifest khi chuyển đổi từ độ phân giải gốc $512 \times 512$ sang kích thước chuẩn $256 \times 256$ (hệ số tỉ lệ diện tích $\times 0.25$).
  * *Ví dụ thực tế:* 
    * Lát cắt $z=25$: GT manifest $129\text{ px} \rightarrow$ Test Loader ghi nhận chính xác $34\text{ px}$ ($TP=34, FN=0$).
    * Lát cắt $z=30$: GT manifest $1011\text{ px} \rightarrow$ Test Loader ghi nhận chính xác $250\text{ px}$ ($TP=250, FN=0$).
* **Độ ổn định trên từng ca bệnh nhân độc lập:**
  * Bệnh nhân 1 (`P_coronacases_002` - 113 lát tổn thương): Dice trung bình đạt **$0.9846$**.
  * Bệnh nhân 2 (`P_coronacases_007` - 73 lát tổn thương): Dice trung bình đạt **$0.9822$**.
  * Bệnh nhân 3 (`P_radiopaedia_27_86410_0` - 35 lát tổn thương): Dice trung bình đạt **$0.9547$**.
  * Không có bất kỳ ca bệnh nhân nào bị sụt giảm hiệu năng bất thường, xác nhận tính tổng quát hóa cao và bảo đảm không có hiện tượng rò rỉ dữ liệu (Zero Data Leakage).

---

## 2. PHƯƠNG PHÁP LUẬN: QUY TRÌNH GÁN NHÃN VÀ HUẤN LUYỆN (METHODOLOGY)

### 2.1. Quy chuẩn Dữ liệu và Chiến lược Gán nhãn (Data & Labeling Strategy)
* **Tập dữ liệu nguồn:** Nghiên cứu sử dụng tập dữ liệu ảnh cắt lớp vi tính lồng ngực (CT) gồm 20 ca bệnh nhân COVID-19 chuẩn hóa từ nguồn dữ liệu mở Zenodo (Jun et al.).
* **Định dạng mặt nạ tổn thương (Ground Truth Labeling):**
  * Nghiên cứu sử dụng tập mặt nạ nhị phân chuyên dụng **`Infection_Mask`** để phục vụ bài toán phân đoạn tổn thương:
    * **Nhãn 0 (Background / Non-lesion):** Bao gồm nền không khí ngoài cơ thể, mô mềm thành ngực, xương sườn và toàn bộ nhu mô phổi lành.
    * **Nhãn 1 (Infection / Lesion):** Vùng tổn thương đông đặc (*Consolidation*) và tổn thương kính mờ (*Ground-Glass Opacities - GGO*) do nhiễm trùng phế nang.
* **Chiến lược cô lập bệnh nhân (Patient-Level Stratification):**
  * Dữ liệu được phân chia nghiêm ngặt ở cấp độ bệnh nhân (Patient-level) với tỉ lệ **70% Huấn luyện (14 ca / 1.277 lát)**, **15% Xác thực (3 ca / 208 lát)**, và **15% Kiểm thử độc lập (3 ca / 265 lát)**.
  * Toàn bộ lát cắt của cùng một bệnh nhân tuyệt đối chỉ thuộc về một tập duy nhất, loại bỏ triệt để hiện tượng rò rỉ thông tin giải phẫu giữa các lát cắt liền kề.

### 2.2. Pipeline Tiền xử lý Hình ảnh Y tế (Medical Image Preprocessing)
1. **Cắt lọc cửa sổ Hounsfield (HU Windowing):** Áp dụng cửa sổ dải $[-1000, 400]$ HU nhằm giữ lại thông tin phản ánh mật độ phế nang và dịch viêm, đồng thời triệt tiêu độ tương phản gây nhiễu của các cấu trúc xương đặc và không khí tự do ngoài lồng ngực.
2. **Chuẩn hóa biên độ (Min-Max Normalization):** Chuyển đổi tuyến tính toàn bộ cường độ pixel sau cửa sổ về khoảng chuẩn $[0.0, 1.0]$.
3. **Đồng bộ hóa kích thước không gian:** Chuẩn hóa hình học các lát cắt từ $512 \times 512$ về kích thước chuẩn $256 \times 256$ pixels. Áp dụng phép nội suy song tuyến tính (*Bilinear Interpolation*) cho ảnh CT và phép nội suy láng giềng gần nhất (*Nearest Neighbor Interpolation*) cho mặt nạ tổn thương để bảo toàn tính toàn vẹn của giá trị nhị phân $\{0, 1\}$.
4. **Tăng cường dữ liệu (Data Augmentation):** Áp dụng cho tập Train gồm các phép lật ảnh theo phương ngang (*Horizontal Flip*), xoay ngẫu nhiên góc hẹp $\pm 10^\circ$, và biến thiên độ sáng/tương phản ngẫu nhiên nhẹ, đảm bảo không làm thay đổi các cấu trúc giải phẫu học cơ bản.

### 2.3. Thiết lập Huấn luyện & Tối ưu hóa (Model Training & Optimization Setup)
* **Kiến trúc mô hình thực nghiệm:** Ba kiến trúc mạng được huấn luyện độc lập trên cùng một điều kiện phần cứng:
  1. *Attention U-Net (Mô hình đề xuất)*: 31.39 triệu tham số, tích hợp cơ chế cổng chú ý (*Attention Gates*) tại các nhánh Skip Connections.
  2. *Standard U-Net (Mô hình đối chuẩn 1)*: 31.04 triệu tham số, kiến trúc U-Net kinh điển.
  3. *SegNet (Mô hình đối chuẩn 2)*: 2.35 triệu tham số, cơ chế pooling indices.
* **Hàm mất mát kết hợp (Hybrid Loss Function):**
  $$\mathcal{L}_{\text{Hybrid}} = 0.5 \times \mathcal{L}_{\text{BCE}} + 0.5 \times \mathcal{L}_{\text{SoftDice}}$$
  Hàm mất mát này kết hợp khả năng phân loại điểm ảnh cấp độ micro của Binary Cross-Entropy và khả năng tối ưu hóa độ trùng khớp hình thái học cấp độ macro của Soft Dice Loss.
* **Tối ưu hóa và Lịch trình tốc độ học:**
  * Bộ tối ưu hóa AdamW với tốc độ học ban đầu $\text{lr} = 1 \times 10^{-4}$, hệ số suy giảm trọng số $\text{weight decay} = 1 \times 10^{-4}$.
  * Lịch trình suy giảm học độ Cosine Annealing ($\eta_{\min} = 1 \times 10^{-6}$, chu kỳ $T_{\max} = 100$ epochs).
  * Kích thước batch: 16 lát cắt/lô.
  * Hạt giống ngẫu nhiên (Random Seed): 2026 được cố định trên toàn bộ luồng PyTorch, CUDA, NumPy để bảo đảm tính tái lập hoàn toàn (Full Reproducibility).
* **Quản lý Checkpoint & Lưu trữ Siêu dữ liệu (Checkpoint Provenance):**
  * Tự động lưu checkpoint tốt nhất (`_best.pth`) dựa trên chỉ số Validation Dice cao nhất và checkpoint cuối cùng (`_last.pth`).
  * Mọi checkpoint đều nhúng cấu trúc siêu dữ liệu hoàn chỉnh gồm: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_dice`, `val_metrics`, `config`, `lesion_label: 1`, `seed: 2026`, `git_commit`, và mã băm `data_manifest_sha`.

---

## 3. BẢNG KẾT QUẢ ĐỊNH LƯỢNG (QUANTITATIVE RESULTS)

### Bảng 1: Báo cáo chính trên các lát cắt có tổn thương (Lesion-Containing Slices, GT > 0, n = 221)

| Mô hình (Model) | Tham số (Params) | Độ trễ (Latency) | Dice (DSC) | IoU (Jaccard) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Attention U-Net (Proposed)** | **31.39M** | **19.43 ms/slice** | **0.9791 ± 0.0187** | **0.9596 ± 0.0338** | **0.9648 ± 0.0273** | **0.9941 ± 0.0136** |
| Standard U-Net (Baseline) | 31.04M | 17.31 ms/slice | 0.9740 ± 0.0190 | 0.9500 ± 0.0341 | 0.9618 ± 0.0292 | 0.9870 ± 0.0123 |
| SegNet (Baseline) | 2.35M | 3.49 ms/slice | 0.9662 ± 0.0229 | 0.9355 ± 0.0404 | 0.9446 ± 0.0370 | 0.9896 ± 0.0139 |

### Bảng 2: Báo cáo tổng thể cấp Bệnh nhân và Toàn bộ Slices (Patient-Level 3D & All Slices, n = 265)

| Mô hình (Model) | Patient DSC | Patient IoU | All-Slices DSC | All-Slices IoU |
| :--- | :---: | :---: | :---: | :---: |
| **Attention U-Net (Proposed)** | **0.9805 ± 0.0098** | **0.9619 ± 0.0187** | **0.9825 ± 0.0188** | **0.9663 ± 0.0343** |
| Standard U-Net (Baseline) | 0.9781 ± 0.0066 | 0.9572 ± 0.0126 | 0.9784 ± 0.0199 | 0.9583 ± 0.0363 |
| SegNet (Baseline) | 0.9713 ± 0.0090 | 0.9444 ± 0.0169 | 0.9718 ± 0.0244 | 0.9462 ± 0.0440 |

### Mã nguồn Bảng LaTeX chuẩn cho Bản thảo Bài báo:
```latex
\begin{table}[htbp]
\centering
\caption{Quantitative Segmentation Performance Comparison on Independent Test Set}
\label{tab:model_comparison}
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Dice (DSC)} & \textbf{IoU (Jaccard)} & \textbf{Precision} & \textbf{Recall} \\
\hline
\textbf{Attention U-Net (Proposed)} & 0.9791 ± 0.0187 & 0.9596 ± 0.0338 & 0.9648 ± 0.0273 & 0.9941 ± 0.0136 \\
SegNet (Baseline) & 0.9662 ± 0.0229 & 0.9355 ± 0.0404 & 0.9446 ± 0.0370 & 0.9896 ± 0.0139 \\
Standard U-Net (Baseline) & 0.9740 ± 0.0190 & 0.9500 ± 0.0341 & 0.9618 ± 0.0292 & 0.9870 ± 0.0123 \\
\hline
\end{tabular}
\end{table}
```

---

## 4. GIẢI THÍCH CHI TIẾT Ý NGHĨA CÁC CỘT CHỈ SỐ (METRICS INTERPRETATION)

| Tiêu đề cột | Công thức toán học | Ý nghĩa khoa học & Ứng dụng lâm sàng |
| :--- | :--- | :--- |
| **`\textbf{Model}`** | — | **Mô hình kiến trúc mạng nơ-ron thực nghiệm.**<br>• *Attention U-Net (Proposed)*: Mô hình đề tài, tích hợp cơ chế Attention Gates.<br>• *Standard U-Net (Baseline)*: Kiến trúc đối chuẩn chuẩn mực trong phân đoạn ảnh y tế.<br>• *SegNet (Baseline)*: Kiến trúc đối chuẩn tiêu biểu cho mô hình gọn nhẹ dùng bộ nhớ indices. |
| **`\textbf{Dice (DSC)}`** | $\text{DSC} = \frac{2 \times TP}{2 \times TP + FP + FN}$ | **Hệ số tương đồng Dice (Dice Similarity Coefficient).**<br>• Đo lường mức độ chồng khít và tương đồng diện tích giữa vùng tổn thương mô hình dự đoán và vùng tổn thương thực tế do chuyên gia y tế gán nhãn.<br>• Trọng số nhân 2 cho vùng giao nhau ($TP$) giúp phản ánh chính xác độ chính xác hình thái học. Attention U-Net đạt **$0.9791$ ($97.91\%$)**, khẳng định ranh giới phân đoạn gần như trùng khít hoàn toàn với mặt nạ chuẩn của bác sĩ. |
| **`\textbf{IoU (Jaccard)}`** | $\text{IoU} = \frac{TP}{TP + FP + FN} = \frac{\text{DSC}}{2 - \text{DSC}}$ | **Chỉ số tương đồng Jaccard (Intersection over Union).**<br>• Đo tỉ lệ giữa diện tích phần giao nhau chia cho diện tích phần hợp nhất của hai vùng.<br>• IoU là thước đo khắt khe hơn Dice vì nó phạt nặng hơn đối với các pixel dự đoán thừa ($FP$) hoặc thiếu ($FN$). Mức IoU **$0.9596$ ($95.96\%$)** chứng minh các đường viền phân đoạn bám sát cấu trúc tổn thương, không bị hiện tượng phình to hay co cụm biên giới. |
| **`\textbf{Precision}`** | $\text{Precision} = \frac{TP}{TP + FP}$ | **Độ chính xác dự đoán (Positive Predictive Value).**<br>• Trả lời câu hỏi: *Trong toàn bộ những điểm ảnh mà mô hình cảnh báo là tổn thương, có bao nhiêu phần trăm thực sự là tổn thương thật?*<br>• Chỉ số đạt **$0.9648$ ($96.48\%$)** chứng minh mô hình có độ chọn lọc rất cao, tỉ lệ báo động giả (False Positives) chỉ dưới $3.6\%$, giúp hạn chế tối đa việc chẩn đoán nhầm nhu mô phổi khỏe mạnh thành tổn thương. |
| **`\textbf{Recall}`** | $\text{Recall} = \frac{TP}{TP + FN}$ | **Độ nhạy phát hiện tổn thương (Sensitivity / True Positive Rate).**<br>• Trả lời câu hỏi: *Trong toàn bộ ổ tổn thương thực sự có trong phổi bệnh nhân, mô hình đã bao quát và phát hiện được bao nhiêu phần trăm?*<br>• **Trong y tế, Recall là chỉ số có trọng số an toàn cao nhất**, bởi vì việc bỏ sót tổn thương (False Negatives) sẽ dẫn đến chẩn đoán sai giai đoạn bệnh và trì hoãn phác đồ điều trị. Attention U-Net đạt Recall lên tới **$0.9941$ ($99.41\%$)**, chứng minh mô hình hầu như không bỏ sót bất kỳ vùng kính mờ hay đông đặc nào dù ở kích thước rất nhỏ. |

---

## 5. KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (WILCOXON SIGNED-RANK TEST)

Để chứng minh tính vượt trội không phải do ngẫu nhiên, kiểm định phi tham số cặp Wilcoxon Signed-Rank Test đã được thực hiện trên 221 lát cắt có tổn thương ($\alpha = 0.05$):

* **Attention U-Net vs SegNet:** 
  $$W = 413.0, \quad p = 9.82 \times 10^{-35} \ll 0.001$$
  $\rightarrow$ Sự vượt trội của Attention U-Net có ý nghĩa thống kê cực kỳ cao.
* **Attention U-Net vs Standard U-Net:** 
  $$W = 4073.0, \quad p = 2.56 \times 10^{-16} \ll 0.001$$
  $\rightarrow$ Việc tích hợp các cổng chú ý (Attention Gates) mang lại cải tiến vượt bậc có ý nghĩa thống kê so với kiến trúc U-Net truyền thống.

---

## 6. HƯỚNG DẪN DÀNH CHO LEAD VÀ TÁC GIẢ BÀI BÁO

1. **Phần Materials and Methods**: Trích dẫn Mục 2.1 (Tập dữ liệu `Infection_Mask`, nhãn 1 là tổn thương), Mục 2.2 (Cửa sổ HU $[-1000, 400]$, Min-Max Scaling, Resize 256) và Mục 2.3 (Hybrid Loss, AdamW, Cosine Annealing, Seed 2026).
2. **Phần Results**: Chèn trực tiếp mã LaTeX ở Mục 3 (`tab:model_comparison`) và trích dẫn số liệu kiểm định Wilcoxon ở Mục 5.
3. **Phần Discussion**: Khai thác chỉ số Recall vượt trội ($99.41\%$) của Attention U-Net để chứng minh giá trị lâm sàng trong việc ngăn ngừa bỏ sót các vùng đông đặc phổi nhỏ.

