::: center
**METRICS & RESEARCH SPECIFICATION:**\
**CT LUNG LESION SEGMENTATION**\

------------------------------------------------------------------------
:::

  ------------- --------------------------------------------------- -- --
  **Đề tài:**   Phân đoạn tổn thương CT phổi bằng Attention U-Net      
  ------------- --------------------------------------------------- -- --

------------------------------------------------------------------------

# Tổng hợp tổng quan tài liệu & Khoảng trống nghiên cứu (Research Gap)

Dựa trên ma trận tổng quan tài liệu (Literature Matrix -- `RES-02`) được
tổng hợp từ 35+ bài báo khoa học giai đoạn 2018--2024 (IEEE, arXiv,
Springer, PubMed):

## Bối cảnh kỹ thuật & Nhược điểm của các kiến trúc Baseline

- **Standard U-Net & SegNet:** Dù là các kiến trúc Encoder-Decoder kinh
  điển mang lại hiệu năng phân đoạn ấn tượng trên ảnh y tế, U-Net truyền
  thống và SegNet tồn tại các hạn chế lớn khi áp dụng vào ảnh CT phổi:

  - **Nhiễu tín hiệu từ Skip Connections:** Các kết nối tắt (skip
    connections) trực tiếp truyền các đặc trưng cấp thấp (low-level
    features) từ Encoder sang Decoder mà không qua bộ lọc, dẫn đến việc
    truyền kèm nhiều nhiễu nền (thành ngực, mô lành, mạch máu xung
    quanh).

  - **Hạn chế đối với tổn thương khó:** Đối với các tổn thương có kích
    thước rất nhỏ ($<5\%$ diện tích phổi) hoặc các vùng tổn thương có
    ranh giới không rõ ràng như tổn thương kính mờ (Ground-Glass Opacity
    -- GGO) và đông đặc (Consolidation), U-Net chuẩn thường bị bỏ sót
    (False Negative cao) hoặc dự đoán tràn biên (False Positive cao).

## Khoảng trống nghiên cứu (Research Gap)

1.  **Khoảng trống 1 (Độ chính xác phân đoạn vùng ranh giới mờ & kích
    thước nhỏ):** Mặc dù các cơ chế Attention Gate (AG) đã được đề xuất
    để lọc nhiễu ở skip connections, việc đánh giá định lượng khả năng
    phân đoạn trên các tổn thương khó (GGO, đông đặc, tổn thương nhỏ)
    trên bộ dữ liệu CT phổi chuẩn hóa theo cấp độ bệnh nhân
    (Patient-level split) vẫn chưa được phân tích đầy đủ và đối chuẩn
    nghiêm ngặt với cả U-Net lẫn SegNet.

2.  **Khoảng trống 2 (Đánh đổi tài nguyên & Hiệu năng tính toán --
    Trade-off):** Đa số các nghiên cứu tập trung tối ưu chỉ số Dice/IoU
    mà thiếu các đánh giá định lượng hệ thống về sự đánh đổi giữa mức
    tăng hiệu năng phân đoạn và chi phí tài nguyên tính toán (số lượng
    tham số/parameters, dung lượng bộ nhớ GPU, thời gian suy
    luận/latency ms per slice). Đây là yếu tố quyết định tính khả thi
    khi triển khai trên các thiết bị hỗ trợ chẩn đoán lâm sàng thời gian
    thực.

# Câu hỏi nghiên cứu (Research Questions -- RQs)

Nhằm giải quyết các khoảng trống nghiên cứu trên, đề tài xác định 2 câu
hỏi nghiên cứu cốt lõi:

- **RQ1 (Độ chính xác phân đoạn & Vùng tổn thương khó):**\
  *Cơ chế Attention Gate khi tích hợp vào U-Net cải thiện độ chính xác
  phân đoạn (đặc biệt đối với tổn thương nhỏ $<5\%$ diện tích phổi và
  tổn thương có ranh giới mờ GGO/Consolidation) như thế nào so với
  Standard U-Net và SegNet?*

- **RQ2 (Đánh đổi tài nguyên & Hiệu năng tính toán):**\
  *Sự đánh đổi (trade-off) giữa hiệu năng phân đoạn (DSC, IoU) và chi
  phí tài nguyên tính toán (số lượng tham số/parameters, dung lượng bộ
  nhớ GPU, thời gian suy luận latency ms/slice) khi thêm các module
  Attention Gate là gì?*

# Định nghĩa toán học bộ Metrics đánh giá phân đoạn (Image Segmentation Metrics)

Để đảm bảo tính khách quan và nhất quán trong suốt quá trình thực nghiệm
(W3 -- Baseline, W4 -- Attention U-Net, W5 -- Evaluation), 4 chỉ số
thống kê toán học sau được thiết lập làm bộ tiêu chí đánh giá chuẩn hóa:

## Dice Similarity Coefficient (DSC / Dice Score)

- **Công thức toán học:** $$\begin{equation}
          \text{DSC} = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}
  \end{equation}$$ Trong đó:

  - $\text{TP}$ (True Positive): Số lượng điểm ảnh tổn thương được mô
    hình dự đoán chính xác.

  - $\text{FP}$ (False Positive): Số lượng điểm ảnh mô lành bị mô hình
    dự đoán nhầm là tổn thương.

  - $\text{FN}$ (False Negative): Số lượng điểm ảnh tổn thương bị mô
    hình bỏ sót.

- **Ý nghĩa lâm sàng:** Đo độ trùng lặp không gian (spatial overlap)
  giữa vùng mask dự đoán và nhãn chuẩn Ground Truth ($[0, 1]$, càng gần
  $1$ càng chính xác).

- **Biến thể liên tục trong Hàm Loss (Soft Dice Loss):**
  $$\begin{equation}
          \mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^{N} p_i g_i + \epsilon}{\sum_{i=1}^{N} p_i^2 + \sum_{i=1}^{N} g_i^2 + \epsilon}
  \end{equation}$$ Với $p_i \in [0, 1]$ là xác suất nhị phân đầu ra
  Sigmoid, $g_i \in \{0, 1\}$ là nhãn Ground Truth, và
  $\epsilon = 10^{-5}$ là hằng số làm mịn (smooth factor) giúp tránh lỗi
  chia cho 0 và nổ gradient.

## Intersection over Union (IoU / Jaccard Index)

- **Công thức toán học:** $$\begin{equation}
          \text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{\text{DSC}}{2 - \text{DSC}}
  \end{equation}$$

- **Ý nghĩa:** Đo tỷ lệ diện tích/thể tích phần giao so với phần hợp
  giữa mask dự đoán và Ground Truth. IoU đưa ra hình phạt nghiêm khắc
  hơn đối với các lỗi phân đoạn so với DSC, giúp đánh giá khắt khe độ
  chính xác ranh giới.

## Precision (Positive Predictive Value -- PPV)

- **Công thức toán học:** $$\begin{equation}
          \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  \end{equation}$$

- **Ý nghĩa:** Đo lường tỷ lệ các điểm ảnh được mô hình gán nhãn "tổn
  thương" thực sự là tổn thương. Precision cao chứng tỏ mô hình ít bị
  cảnh báo nhầm (kiểm soát False Positive), hạn chế gây hoang mang cho
  bác sĩ chẩn đoán.

## Recall / Sensitivity (True Positive Rate -- TPR)

- **Công thức toán học:** $$\begin{equation}
          \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  \end{equation}$$

- **Ý nghĩa:** Đo lường khả năng bắt trọn toàn bộ vùng tổn thương có
  trong ảnh CT. Recall cao đồng nghĩa với việc mô hình ít bỏ sót bệnh
  (kiểm soát False Negative). Trong ứng dụng y tế, Recall đạt mức cao là
  ưu tiên hàng đầu để tránh bỏ qua các ca bệnh nguy hiểm.

# Định nghĩa chỉ số tài nguyên tính toán (Computational Metrics cho RQ2)

Để giải quyết và đo lường định lượng câu hỏi nghiên cứu RQ2, bộ chỉ số
tài nguyên tính toán được chuẩn hóa như sau:

## Số lượng tham số (Trainable Parameters -- $M$)

- **Định nghĩa:** Tổng số lượng trọng số (weights) và độ lệch (biases)
  có thể cập nhật trong quá trình huấn luyện mạng.

- **Đơn vị đo:** Triệu tham số (Millions -- $M$).

- **Cách đo trong PyTorch:** $$\begin{equation}
          \text{Trainable Params} = \sum_{\theta \in \Theta} \text{numel}(\theta) \quad \text{với } \theta.\text{requires\_grad} = \text{True}
  \end{equation}$$

## Độ phức tạp tính toán (FLOPs / GFLOPs)

- **Định nghĩa:** Tổng số phép tính số thực (Floating Point Operations)
  cần thiết để thực hiện một lượt lan truyền tiến (forward pass) cho 1
  lát cắt 2D kích thước chuẩn ($256 \times 256$ hoặc $512 \times 512$).

- **Đơn vị đo:** Giga FLOPs ($1\,\text{GFLOP} = 10^9\,\text{FLOPs}$).

- **Công cụ đo:** Sử dụng thư viện `thop` (`thop.profile`) hoặc
  `ptflops` với đầu vào tensor 1 slice $(1, 1, H, W)$.

## Thời gian suy luận (Inference Latency -- $\text{ms/slice}$)

- **Định nghĩa:** Thời gian trung bình để mô hình xử lý suy luận hoàn
  tất 1 lát cắt 2D.

- **Điều kiện đo bắt buộc:**

  - Đo trên cùng một cấu hình phần cứng cố định (ví dụ: GPU NVIDIA T4 /
    V100 / RTX 3090).

  - Đặt $\text{batch\_size} = 1$ để mô phỏng môi trường suy luận thực tế
    thời gian thực.

  - Thực hiện 100 warm-up runs trước khi đo, sau đó lấy trung bình thời
    gian thực thi qua 1.000 slices kiểm thử bằng đồng hồ GPU chuẩn
    (`torch.cuda.Event`).

- **Đơn vị đo:** Milliseconds per slice ($\text{ms/slice}$).

## Bộ nhớ GPU đỉnh (Peak GPU VRAM Usage -- MB/GB)

- **Định nghĩa:** Dung lượng bộ nhớ VRAM lớn nhất mà mô hình chiếm dụng
  trong quá trình suy luận (Inference phase).

- **Cách đo trong PyTorch:** Sử dụng hàm
  `torch.cuda.max_memory_allocated()` sau khi chạy suy luận một
  batch/volume.

- **Đơn vị đo:** Megabytes (MB) hoặc Gigabytes (GB).

# Quy định cấp độ tính toán (Slice-level vs. Patient-level)

Để đảm bảo sự nhất quán chặt chẽ giữa các thành viên phụ trách huấn
luyện (Hải, Hậu) và viết bài báo (Duy), quy định cấp độ đánh giá được
thiết lập như sau:

## Cấp độ đánh giá chính (Primary Metric -- Slice-level Macro Average)

Tất cả các chỉ số phân đoạn
($\text{DSC}, \text{IoU}, \text{Precision}, \text{Recall}$) sẽ được tính
toán độc lập cho từng lát cắt 2D $i$. Chỉ số tổng hợp của toàn tập kiểm
thử là trung bình cộng (Macro Average) trên tất cả các lát cắt 2D trong
tập Test: $$\begin{equation}
    \text{DSC}_{\text{slice\_avg}} = \frac{1}{N} \sum_{i=1}^{N} \text{DSC}_i
\end{equation}$$ Quy chuẩn này phù hợp hoàn toàn với bản chất huấn luyện
2D của U-Net, Attention U-Net và SegNet.

## Cấp độ đánh giá bổ trợ (Secondary Metric -- Patient-level / Volume-level Aggregation)

Gom nhóm các lát cắt 2D theo từng bệnh nhân $k$ để tái tạo thể tích tổn
thương 3D. Tính toán $\text{DSC}_{\text{volume}}$ trên thể tích 3D gộp
của từng ca bệnh: $$\begin{equation}
    \text{DSC}_{\text{patient\_}k} = \frac{2 \cdot \sum_{i \in \text{Patient}_k} \text{TP}_i}{2 \cdot \sum_{i \in \text{Patient}_k} \text{TP}_i + \sum_{i \in \text{Patient}_k} \text{FP}_i + \sum_{i \in \text{Patient}_k} \text{FN}_i}
\end{equation}$$ Giúp đánh giá tác động lâm sàng ở mức độ từng ca bệnh
thực tế.

## Yêu cầu đối với Script đánh giá (EVAL-03 đến EVAL-06)

Các script Python đánh giá bắt buộc phải tự động xuất ra 2 bảng kết quả
riêng biệt:

- `test_metrics_slice_level.csv`: Kết quả theo từng slice và trung bình
  toàn tập.

- `test_metrics_patient_level.csv`: Kết quả tổng hợp theo từng ID bệnh
  nhân.

# Quy ước xử lý trường hợp biên (Edge Case Handling)

Trong dữ liệu CT phổi, tồn tại nhiều lát cắt thuộc vùng phổi lành không
chứa tổn thương ($\text{GT} = 0$). Để tránh lỗi toán học chia cho 0
($\frac{0}{0}$) và hiện tượng sai lệch điểm số, quy ước kỹ thuật sau
được áp dụng thống nhất:

## Quy ước toán học cho 3 kịch bản Edge Case

  **Kịch bản**                                 **GT**         **Pred**             **Tình trạng TP, FP, FN**          **DSC**   **IoU**   **Prec.**   **Recall**
  --------------------------------------- ---------------- --------------- ----------------------------------------- --------- --------- ----------- ------------
  1\. Dự đoán đúng slice âm tính             $0$ (Rỗng)      $0$ (Rỗng)        $\text{TP}=\text{FP}=\text{FN}=0$       $1.0$     $1.0$      $1.0$       $1.0$
  2\. Cảnh báo nhầm trên slice âm tính       $0$ (Rỗng)     $>0$ (Có vết)   $\text{TP}=0, \text{FP}>0, \text{FN}=0$    $0.0$     $0.0$      $0.0$      $1.0^*$
  3\. Bỏ sót hoàn toàn slice dương tính    $>0$ (Có bệnh)    $0$ (Rỗng)     $\text{TP}=0, \text{FP}=0, \text{FN}>0$    $0.0$     $0.0$      $0.0$       $0.0$

  : Quy ước tính toán cho các trường hợp biên. (\*Lưu ý: Ở kịch bản 2,
  Recall quy ước tránh chia cho 0 hoặc có thể gán `NaN` khi tính toán
  độc lập).

## Chiến lược báo cáo song song (Dual-Reporting Strategy)

Đội ngũ kỹ thuật (Hải, Hậu) khi chạy các script đánh giá sẽ báo cáo kết
quả theo 2 chế độ:

- **Chế độ 1 (Lesion-containing Slices -- $\text{GT} > 0$) \[BÁO CÁO
  CHÍNH IN BÀI BÁO\]:** Chỉ tính toán metrics trên các lát cắt thực sự
  chứa tổn thương ($\text{GT} > 0$). Chế độ này phản ánh chính xác năng
  lực định vị và phân đoạn vùng bệnh của mô hình mà không bị ảnh hưởng
  bởi số lượng lớn các slice rỗng.

- **Chế độ 2 (All Slices -- Bao gồm cả $\text{GT} = 0$) \[BÁO CÁO BỔ
  TRỢ\]:** Áp dụng quy ước bảng trên cho toàn bộ lát cắt trong tập kiểm
  thử.

# Ma trận tương quan & Phiên bản hóa sử dụng

  **Metric / Nhóm**      **Chỉ số cụ thể**                                **Vai trò & Ứng dụng trong Dự án**                                                                                       **Task liên quan**
  ---------------------- ------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------ ---------------------------
  **Segmentation**       DSC, IoU, Precision, Recall                      Huấn luyện (Soft Dice Loss) & Bảng so sánh hiệu năng phân đoạn W5                                                        `MODEL-02`, `EVAL-03--06`
  **Loss Function**      Soft Dice + BCE Hybrid Loss                      Hàm mục tiêu: $\mathcal{L}_{\text{hybrid}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$   `MODEL-04`
  **Computational**      Params (M), FLOPs (G), Latency (ms), VRAM (MB)   Đánh giá sự đánh đổi tài nguyên phục vụ câu hỏi nghiên cứu RQ2                                                           `EVAL-08`
  **Evaluation Level**   Slice-level vs. Patient-level                    Báo cáo đa góc độ (2D slice & 3D patient volume)                                                                         `EVAL-03--08`
  **Edge Case**          Lesion-containing vs. All-slices                 Áp dụng quy ước $\text{GT}=\text{Pred}=0 \Rightarrow \text{DSC}=1.0$; báo cáo song song                                  `EVAL-03--06`

  : Ma trận tương quan giữa bộ tiêu chuẩn đánh giá và các nhiệm vụ thực
  nghiệm.

# Quy trình đánh giá thực nghiệm (Evaluation Pipeline W5)

1.  Tất cả 4 chỉ số phân đoạn và 4 chỉ số tài nguyên sẽ được tính toán
    độc lập trên tập kiểm thử **Independent Test Set (15%
    Patient-level)** \[`EVAL-01`, `EVAL-02`\].

2.  Báo cáo kết quả dưới dạng **Giá trị trung bình $\pm$ Độ lệch chuẩn
    (Mean $\pm$ Std)** kèm khoảng tin cậy 95% (95% Confidence Interval)
    \[`EVAL-03` -- `EVAL-06`\].

3.  Tiến hành kiểm định thống kê cặp **Wilcoxon signed-rank test**
    ($p < 0.05$) để xác nhận sự cải thiện của Attention U-Net có ý nghĩa
    thống kê so với U-Net và SegNet \[`EVAL-07`\].

4.  Tổng hợp bảng so sánh hiệu năng -- tài nguyên (**Trade-off Matrix**)
    kết hợp đồ thị Radar Chart để trả lời trọn vẹn RQ1 và RQ2
    \[`EVAL-08`\].
