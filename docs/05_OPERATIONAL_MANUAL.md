# 05_OPERATIONAL_MANUAL.md

## 1. Mục đích

Quy định cách nhóm phối hợp và vận hành trong suốt 8 tuần, đảm bảo công việc được cập nhật, code được kiểm soát và rủi ro được xử lý kịp thời.

> **Cấu trúc:** Phần A — quy trình vận hành, Git Flow, xử lý rủi ro; Phần B — Metrics & Research Specification (gộp từ file `Metrics_Specification.md`).

---

# PHẦN A — VẬN HÀNH & QUY TRÌNH LÀM VIỆC

## 2. Weekly Sync

Mỗi thành viên báo cáo ngắn:

1. Đã hoàn thành gì?
2. Đang làm gì?
3. Kết quả hiện tại?
4. Có vấn đề hoặc blocker nào?
5. Tuần tiếp theo sẽ làm gì?

### Nguyên tắc

- Đi đúng trọng tâm.
- Mỗi thành viên phải có cập nhật.
- Blocker phải được nêu rõ, không để tồn đọng.
- Sau cuộc họp phải thống nhất các task tiếp theo.

---

## 3. Kênh trao đổi

### Trao đổi nhanh

Dùng nhóm chat chung để:

- Thông báo tiến độ.
- Trao đổi vấn đề ngắn.
- Báo blocker.
- Thống nhất các quyết định nhanh.

### Công việc chính thức

Các nội dung quan trọng phải được ghi nhận trong:

- GitHub Issues / Project.
- Pull Request.
- Markdown Documentation.
- Experiment Log.

> Không để thông tin quan trọng chỉ tồn tại trong tin nhắn cá nhân.

---

## 4. Quy trình làm việc

Mỗi task thực hiện theo quy trình:

```text
Task
 ↓
Code / Research
 ↓
Test
 ↓
Commit
 ↓
Pull Request
 ↓
Review
 ↓
Merge
 ↓
Update Documentation
```

Task chỉ được xem là hoàn thành khi có kết quả và được cập nhật vào tài liệu liên quan.

---

## 5. Git Flow

### Nhánh chính

```text
main
```

`main` chỉ chứa code đã được kiểm tra và có thể sử dụng.

### Nhánh tính năng

Đặt tên theo:

```text
feat/<ten-tinh-nang>
```

Ví dụ:

```text
feat/unet-training
feat/attention-unet
feat/dataset-loader
feat/segnet
feat/streamlit-demo
```

### Commit

Commit nên ngắn và thể hiện rõ thay đổi:

```text
feat: add attention gate
fix: correct mask preprocessing
docs: update experiment guide
refactor: improve dataset loader
```

---

## 6. Pull Request

Không merge trực tiếp vào `main`.

Quy trình:

```text
Create Branch
     ↓
Implement
     ↓
Push
     ↓
Create Pull Request
     ↓
Review
     ↓
Fix if needed
     ↓
Merge
```

PR cần có:

- Mục đích thay đổi.
- Nội dung chính.
- Cách kiểm tra.
- Kết quả nếu có.
- Issue/Task liên quan nếu có.

---

## 7. Code Review

Người review kiểm tra tối thiểu:

- Code chạy được.
- Không phá chức năng hiện tại.
- Không chứa file hoặc dữ liệu không cần thiết.
- Tên biến/hàm dễ hiểu.
- Không hard-code thông tin quan trọng.
- Kết quả thực nghiệm có thể truy xuất nếu PR liên quan đến Experiment.

Sau khi review xong mới merge.

---

## 8. Xử lý rủi ro

### 8.1 Hết quota GPU Colab

Ưu tiên:

1. Tiết kiệm GPU bằng cách giảm thời gian thử nghiệm không cần thiết.
2. Chạy preprocessing/visualization trên CPU.
3. Sử dụng checkpoint để tiếp tục training.
4. Chuyển sang môi trường GPU khác nếu có.
5. Điều chỉnh batch size hoặc input size khi cần.

Không thay đổi điều kiện thực nghiệm chính thức chỉ để đối phó với giới hạn tài nguyên mà không ghi nhận thay đổi.

---

### 8.2 Mô hình không hội tụ

Kiểm tra theo thứ tự:

```text
Dataset
 ↓
Mask
 ↓
Preprocessing
 ↓
Loss
 ↓
Learning Rate
 ↓
Model
 ↓
Training Configuration
```

Nếu vẫn không hội tụ:

- Tạo Experiment mới.
- Ghi lại nguyên nhân nghi ngờ.
- Thay đổi từng yếu tố một.
- Không ghi đè kết quả cũ.

---

### 8.3 Kết quả không như kỳ vọng

Không tự ý xóa hoặc bỏ kết quả.

Thực hiện:

1. Kiểm tra lại Dataset và Split.
2. Kiểm tra preprocessing.
3. Kiểm tra training log.
4. Kiểm tra prediction.
5. Phân tích failure case.
6. Thực hiện Experiment bổ sung nếu cần.
7. Ghi nhận kết quả trong Experiment Log.

Kết quả không tốt vẫn là **evidence** nếu thí nghiệm được thực hiện đúng quy trình.

---

## 9. Khi có Blocker

Khi gặp blocker không thể tự giải quyết:

```text
Phát hiện vấn đề
      ↓
Ghi rõ vấn đề
      ↓
Thông báo nhóm
      ↓
Xác định người hỗ trợ
      ↓
Xử lý
      ↓
Cập nhật trạng thái
```

Không để blocker kéo dài mà không thông báo.

---

## 10. Quy tắc chung

- Mỗi thành viên chịu trách nhiệm với task được giao.
- Thay đổi quan trọng phải được ghi nhận.
- Không merge code chưa được review.
- Không xóa hoặc ghi đè evidence của Experiment cũ.
- Không đưa kết quả chưa kiểm chứng vào Paper.
- Khi có thay đổi ảnh hưởng đến nghiên cứu, phải cập nhật tài liệu liên quan.

### Nguyên tắc vận hành

> **Trao đổi rõ → Làm việc có quy trình → Review trước khi merge → Ghi nhận đầy đủ → Không mất Evidence.**

---

# PHẦN B — TỔNG HỢP METRICS & RESEARCH SPECIFICATION

Nội dung bảo toàn nguyên văn từ file `Metrics_Specification.md` cũ (bản export pandoc; phụ trách RES-02, RQ1-RQ2, EVAL).

::: center
**METRICS & RESEARCH SPECIFICATION:**\
**CT LUNG LESION SEGMENTATION**\

------------------------------------------------------------------------
:::

  ------------- --------------------------------------------------- -- --
  **Đề tài:**   Phân đoạn tổn thương CT phổi bằng Attention U-Net      
  ------------- --------------------------------------------------- -- --

------------------------------------------------------------------------

## B.1. Tổng hợp tổng quan tài liệu & Khoảng trống nghiên cứu (Research Gap)

Dựa trên ma trận tổng quan tài liệu (Literature Matrix -- `RES-02`) được
tổng hợp từ 35+ bài báo khoa học giai đoạn 2018--2024 (IEEE, arXiv,
Springer, PubMed):

### B.1.1. Bối cảnh kỹ thuật & Nhược điểm của các kiến trúc Baseline

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

### B.1.2. Khoảng trống nghiên cứu (Research Gap)

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

## B.2. Câu hỏi nghiên cứu (Research Questions -- RQs)

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

## B.3. Định nghĩa toán học bộ Metrics đánh giá phân đoạn (Image Segmentation Metrics)

Để đảm bảo tính khách quan và nhất quán trong suốt quá trình thực nghiệm
(W3 -- Baseline, W4 -- Attention U-Net, W5 -- Evaluation), 4 chỉ số
thống kê toán học sau được thiết lập làm bộ tiêu chí đánh giá chuẩn hóa:

### B.3.1 Dice Similarity Coefficient (DSC / Dice Score)

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

### B.3.2 Intersection over Union (IoU / Jaccard Index)

- **Công thức toán học:** $$\begin{equation}
          \text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{\text{DSC}}{2 - \text{DSC}}
  \end{equation}$$

- **Ý nghĩa:** Đo tỷ lệ diện tích/thể tích phần giao so với phần hợp
  giữa mask dự đoán và Ground Truth. IoU đưa ra hình phạt nghiêm khắc
  hơn đối với các lỗi phân đoạn so với DSC, giúp đánh giá khắt khe độ
  chính xác ranh giới.

### B.3.3 Precision (Positive Predictive Value -- PPV)

- **Công thức toán học:** $$\begin{equation}
          \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  \end{equation}$$

- **Ý nghĩa:** Đo lường tỷ lệ các điểm ảnh được mô hình gán nhãn "tổn
  thương" thực sự là tổn thương. Precision cao chứng tỏ mô hình ít bị
  cảnh báo nhầm (kiểm soát False Positive), hạn chế gây hoang mang cho
  bác sĩ chẩn đoán.

### B.3.4 Recall / Sensitivity (True Positive Rate -- TPR)

- **Công thức toán học:** $$\begin{equation}
          \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  \end{equation}$$

- **Ý nghĩa:** Đo lường khả năng bắt trọn toàn bộ vùng tổn thương có
  trong ảnh CT. Recall cao đồng nghĩa với việc mô hình ít bỏ sót bệnh
  (kiểm soát False Negative). Trong ứng dụng y tế, Recall đạt mức cao là
  ưu tiên hàng đầu để tránh bỏ qua các ca bệnh nguy hiểm.

## B.4. Định nghĩa chỉ số tài nguyên tính toán (Computational Metrics cho RQ2)

Để giải quyết và đo lường định lượng câu hỏi nghiên cứu RQ2, bộ chỉ số
tài nguyên tính toán được chuẩn hóa như sau:

### B.4.1 Số lượng tham số (Trainable Parameters -- $M$)

- **Định nghĩa:** Tổng số lượng trọng số (weights) và độ lệch (biases)
  có thể cập nhật trong quá trình huấn luyện mạng.

- **Đơn vị đo:** Triệu tham số (Millions -- $M$).

- **Cách đo trong PyTorch:** $$\begin{equation}
          \text{Trainable Params} = \sum_{\theta \in \Theta} \text{numel}(\theta) \quad \text{với } \theta.\text{requires\_grad} = \text{True}
  \end{equation}$$

### B.4.2 Độ phức tạp tính toán (FLOPs / GFLOPs)

- **Định nghĩa:** Tổng số phép tính số thực (Floating Point Operations)
  cần thiết để thực hiện một lượt lan truyền tiến (forward pass) cho 1
  lát cắt 2D kích thước chuẩn ($256 \times 256$ hoặc $512 \times 512$).

- **Đơn vị đo:** Giga FLOPs ($1\,\text{GFLOP} = 10^9\,\text{FLOPs}$).

- **Công cụ đo:** Sử dụng thư viện `thop` (`thop.profile`) hoặc
  `ptflops` với đầu vào tensor 1 slice $(1, 1, H, W)$.

### B.4.3 Thời gian suy luận (Inference Latency -- $\text{ms/slice}$)

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

### B.4.4 Bộ nhớ GPU đỉnh (Peak GPU VRAM Usage -- MB/GB)

- **Định nghĩa:** Dung lượng bộ nhớ VRAM lớn nhất mà mô hình chiếm dụng
  trong quá trình suy luận (Inference phase).

- **Cách đo trong PyTorch:** Sử dụng hàm
  `torch.cuda.max_memory_allocated()` sau khi chạy suy luận một
  batch/volume.

- **Đơn vị đo:** Megabytes (MB) hoặc Gigabytes (GB).

## B.5. Quy định cấp độ tính toán (Slice-level vs. Patient-level)

Để đảm bảo sự nhất quán chặt chẽ giữa các thành viên phụ trách huấn
luyện (Hải, Hậu) và viết bài báo (Duy), quy định cấp độ đánh giá được
thiết lập như sau:

### B.5.1 Cấp độ đánh giá chính (Primary Metric -- Slice-level Macro Average)

Tất cả các chỉ số phân đoạn
($\text{DSC}, \text{IoU}, \text{Precision}, \text{Recall}$) sẽ được tính
toán độc lập cho từng lát cắt 2D $i$. Chỉ số tổng hợp của toàn tập kiểm
thử là trung bình cộng (Macro Average) trên tất cả các lát cắt 2D trong
tập Test: $$\begin{equation}
    \text{DSC}_{\text{slice\_avg}} = \frac{1}{N} \sum_{i=1}^{N} \text{DSC}_i
\end{equation}$$ Quy chuẩn này phù hợp hoàn toàn với bản chất huấn luyện
2D của U-Net, Attention U-Net và SegNet.

### B.5.2 Cấp độ đánh giá bổ trợ (Secondary Metric -- Patient-level / Volume-level Aggregation)

Gom nhóm các lát cắt 2D theo từng bệnh nhân $k$ để tái tạo thể tích tổn
thương 3D. Tính toán $\text{DSC}_{\text{volume}}$ trên thể tích 3D gộp
của từng ca bệnh: $$\begin{equation}
    \text{DSC}_{\text{patient\_}k} = \frac{2 \cdot \sum_{i \in \text{Patient}_k} \text{TP}_i}{2 \cdot \sum_{i \in \text{Patient}_k} \text{TP}_i + \sum_{i \in \text{Patient}_k} \text{FP}_i + \sum_{i \in \text{Patient}_k} \text{FN}_i}
\end{equation}$$ Giúp đánh giá tác động lâm sàng ở mức độ từng ca bệnh
thực tế.

### B.5.3 Yêu cầu đối với Script đánh giá (EVAL-03 đến EVAL-06)

Các script Python đánh giá bắt buộc phải tự động xuất ra 2 bảng kết quả
riêng biệt:

- `test_metrics_slice_level.csv`: Kết quả theo từng slice và trung bình
  toàn tập.

- `test_metrics_patient_level.csv`: Kết quả tổng hợp theo từng ID bệnh
  nhân.

## B.6. Quy ước xử lý trường hợp biên (Edge Case Handling)

Trong dữ liệu CT phổi, tồn tại nhiều lát cắt thuộc vùng phổi lành không
chứa tổn thương ($\text{GT} = 0$). Để tránh lỗi toán học chia cho 0
($\frac{0}{0}$) và hiện tượng sai lệch điểm số, quy ước kỹ thuật sau
được áp dụng thống nhất:

### B.6.1 Quy ước toán học cho 3 kịch bản Edge Case

  **Kịch bản**                                 **GT**         **Pred**             **Tình trạng TP, FP, FN**          **DSC**   **IoU**   **Prec.**   **Recall**
  --------------------------------------- ---------------- --------------- ----------------------------------------- --------- --------- ----------- ------------
  1\. Dự đoán đúng slice âm tính             $0$ (Rỗng)      $0$ (Rỗng)        $\text{TP}=\text{FP}=\text{FN}=0$       $1.0$     $1.0$      $1.0$       $1.0$
  2\. Cảnh báo nhầm trên slice âm tính       $0$ (Rỗng)     $>0$ (Có vết)   $\text{TP}=0, \text{FP}>0, \text{FN}=0$    $0.0$     $0.0$      $0.0$      $1.0^*$
  3\. Bỏ sót hoàn toàn slice dương tính    $>0$ (Có bệnh)    $0$ (Rỗng)     $\text{TP}=0, \text{FP}=0, \text{FN}>0$    $0.0$     $0.0$      $0.0$       $0.0$

  : Quy ước tính toán cho các trường hợp biên. (\*Lưu ý: Ở kịch bản 2,
  Recall quy ước tránh chia cho 0 hoặc có thể gán `NaN` khi tính toán
  độc lập).

### B.6.2 Chiến lược báo cáo song song (Dual-Reporting Strategy)

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

## B.7. Ma trận tương quan & Phiên bản hóa sử dụng

  **Metric / Nhóm**      **Chỉ số cụ thể**                                **Vai trò & Ứng dụng trong Dự án**                                                                                       **Task liên quan**
  ---------------------- ------------------------------------------------ ------------------------------------------------------------------------------------------------------------------------ ---------------------------
  **Segmentation**       DSC, IoU, Precision, Recall                      Huấn luyện (Soft Dice Loss) & Bảng so sánh hiệu năng phân đoạn W5                                                        `MODEL-02`, `EVAL-03--06`
  **Loss Function**      Soft Dice + BCE Hybrid Loss                      Hàm mục tiêu: $\mathcal{L}_{\text{hybrid}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$   `MODEL-04`
  **Computational**      Params (M), FLOPs (G), Latency (ms), VRAM (MB)   Đánh giá sự đánh đổi tài nguyên phục vụ câu hỏi nghiên cứu RQ2                                                           `EVAL-08`
  **Evaluation Level**   Slice-level vs. Patient-level                    Báo cáo đa góc độ (2D slice & 3D patient volume)                                                                         `EVAL-03--08`
  **Edge Case**          Lesion-containing vs. All-slices                 Áp dụng quy ước $\text{GT}=\text{Pred}=0 \Rightarrow \text{DSC}=1.0$; báo cáo song song                                  `EVAL-03--06`

  : Ma trận tương quan giữa bộ tiêu chuẩn đánh giá và các nhiệm vụ thực
  nghiệm.

## B.8. Quy trình đánh giá thực nghiệm (Evaluation Pipeline W5)

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

# 06_PRESENTATION_DEFENSE_KIT.md

## 1. Mục đích

Tài liệu chuẩn bị cho phần **thuyết trình và bảo vệ trước hội đồng**.

> **Cấu trúc:** Phần A — cấu trúc slide, phản biện và checklist bảo vệ; Phần B — Tech Note phân tích kiến trúc U-Net / Attention U-Net / SegNet (RES-03, gộp từ file `Tech_Note_UNet_Architecture.md`).

Mục tiêu:

- Trình bày nghiên cứu ngắn gọn, dễ hiểu.
- Ưu tiên sơ đồ, hình ảnh và biểu đồ.
- Chuẩn bị trước các câu hỏi phản biện.
- Đảm bảo mọi nhận định đều có Evidence.

---

# PHẦN A — CẤU TRÚC SLIDE, PHẢN BIỆN & CHECKLIST BẢO VỆ

## 2. Cấu trúc Slide

Khuyến nghị **15–20 slide**.

| #   | Nội dung                    |
| --- | --------------------------- |
| 1   | Title                       |
| 2   | Bối cảnh & vấn đề           |
| 3   | Mục tiêu nghiên cứu         |
| 4   | Research Questions          |
| 5   | Tổng quan phương pháp       |
| 6   | Dataset                     |
| 7   | Preprocessing               |
| 8   | Kiến trúc U-Net             |
| 9   | Attention U-Net             |
| 10  | Attention Gate              |
| 11  | SegNet                      |
| 12  | Training Pipeline           |
| 13  | Evaluation Metrics          |
| 14  | Kết quả định lượng          |
| 15  | Kết quả trực quan           |
| 16  | Failure Cases               |
| 17  | Phân tích RQ1               |
| 18  | Phân tích RQ2               |
| 19  | Demo / Ứng dụng             |
| 20  | Kết luận & hướng phát triển |

Có thể giảm số slide nếu nội dung thực tế không cần đủ 20 slide.

---

## 3. Nguyên tắc thiết kế Slide

### Ưu tiên

```text
Sơ đồ
Hình ảnh
Biểu đồ
Bảng ngắn
→
Text
```

Không đưa nguyên đoạn văn dài lên slide.

### Mỗi slide

- Một thông điệp chính.
- Tiêu đề rõ ràng.
- Chỉ giữ các số liệu quan trọng.
- Font đủ lớn để trình chiếu.
- Hình ảnh phải có chú thích khi cần.

---

## 4. Các hình ảnh quan trọng

Bộ slide nên có tối thiểu:

### Pipeline

```text
CT Scan
  ↓
Preprocessing
  ↓
Model
  ↓
Segmentation
  ↓
Evaluation
```

### Model Architecture

Thể hiện trực quan:

```text
U-Net
Attention U-Net + Attention Gate
SegNet
```

### Kết quả

Nên trình bày:

```text
Original CT | Ground Truth | Prediction
```

### Biểu đồ

Ưu tiên biểu đồ so sánh:

- Dice.
- IoU.
- Precision.
- Recall.

Không đưa quá nhiều biểu đồ nếu không phục vụ RQ.

---

## 5. Chuẩn bị phản biện

## Q1. Vì sao sử dụng ảnh CT 2D thay vì xử lý CT 3D?

**Luận điểm trả lời:**

Nghiên cứu lựa chọn 2D CT slices để phù hợp với phạm vi, tài nguyên tính toán và thời gian thực hiện của đề tài. Cách tiếp cận này cho phép tập trung đánh giá khả năng phân đoạn của các kiến trúc U-Net, Attention U-Net và SegNet trên từng lát cắt.

Cần nêu rõ đây là **giới hạn phạm vi nghiên cứu**, không khẳng định 2D tốt hơn 3D.

---

## Q2. Vì sao chọn Attention U-Net?

**Luận điểm trả lời:**

Attention U-Net được lựa chọn để nghiên cứu tác động của **Attention Gate** trong quá trình phân đoạn. Cơ chế này giúp mô hình tập trung vào các đặc trưng liên quan đến vùng tổn thương thay vì sử dụng toàn bộ đặc trưng một cách đồng đều.

Trong nghiên cứu, hiệu quả của Attention Gate được kiểm chứng bằng thực nghiệm thông qua so sánh với Standard U-Net.

---

## Q3. Attention Gate có thực sự cải thiện kết quả không?

Không trả lời dựa trên giả định.

Trình bày:

1. Kết quả U-Net.
2. Kết quả Attention U-Net.
3. Cùng điều kiện thực nghiệm.
4. So sánh Dice/IoU và các metrics liên quan.
5. Phân tích trường hợp cải thiện hoặc thất bại.

> Kết luận phải dựa trên Evidence của Experiment.

---

## Q4. Vì sao chọn U-Net làm baseline?

U-Net phù hợp với bài toán semantic segmentation và là baseline chính để đánh giá ảnh hưởng của Attention Gate.

So sánh:

```text
U-Net
   ↓
Thêm Attention Gate
   ↓
Attention U-Net
```

Điều này giúp việc phân tích RQ1 rõ ràng hơn.

---

## Q5. Vì sao cần SegNet?

SegNet được sử dụng như một kiến trúc đối chứng để mở rộng việc so sánh ngoài nhóm U-Net.

Kết quả SegNet được đánh giá dưới cùng framework metrics và điều kiện thực nghiệm nhằm phục vụ RQ2.

---

## Q6. Làm thế nào tránh Data Leakage?

Dữ liệu được chia theo **patient/volume**, không chia độc lập từng slice.

Một patient/volume không được xuất hiện đồng thời trong Train, Validation và Test.

---

## Q7. Nếu mô hình cho kết quả không tốt thì sao?

Không loại bỏ kết quả chỉ vì kết quả không tốt.

Nhóm sẽ:

- Kiểm tra pipeline.
- Kiểm tra dữ liệu và mask.
- Phân tích Failure Case.
- Ghi nhận Experiment.
- Đưa giới hạn vào Discussion nếu cần.

Kết quả không như kỳ vọng vẫn có giá trị nếu Experiment được thực hiện đúng quy trình.

---

## Q8. Kết quả có thể áp dụng trực tiếp cho chẩn đoán y tế không?

Không.

Đề tài xây dựng mô hình nhằm **hỗ trợ nghiên cứu và phân đoạn tổn thương**, không thay thế bác sĩ hoặc đưa ra chẩn đoán lâm sàng.

Demo chỉ nhằm minh họa khả năng ứng dụng của mô hình.

---

## 6. Checklist trước bảo vệ

### Nội dung

- [ ] RQs rõ ràng.
- [ ] Dataset được giải thích.
- [ ] Pipeline có sơ đồ.
- [ ] Kiến trúc model có hình minh họa.
- [ ] Metrics được giải thích.
- [ ] Kết quả có Evidence.
- [ ] RQ1 được trả lời.
- [ ] RQ2 được trả lời.
- [ ] Failure Cases được chuẩn bị.
- [ ] Giới hạn nghiên cứu được nêu rõ.

### Demo

- [ ] Model hoạt động.
- [ ] Upload CT hoạt động.
- [ ] Prediction hiển thị đúng.
- [ ] Có mẫu test dự phòng.
- [ ] Có phương án khi demo gặp lỗi.

### Thuyết trình

- [ ] Mỗi thành viên nắm phần của mình.
- [ ] Có người kiểm tra slide cuối.
- [ ] Có bản PDF dự phòng.
- [ ] Có backup source code và model.
- [ ] Đã luyện tập với giới hạn thời gian thực tế.

---

## 7. Nguyên tắc trả lời hội đồng

Khi gặp câu hỏi:

```text
Hiểu câu hỏi
   ↓
Trả lời trực tiếp
   ↓
Đưa Evidence
   ↓
Nêu giới hạn nếu có
```

Không tranh luận dựa trên cảm tính và không khẳng định vượt quá kết quả nghiên cứu.

> **Slide để truyền đạt — Evidence để chứng minh — Phản biện để làm rõ.**

---

# PHẦN B — TECH NOTE: PHÂN TÍCH VÀ ĐỐI CHUẨN KIẾN TRÚC U-NET, ATTENTION U-NET VÀ SEGNET (mã công việc RES-03)

* **Mã công việc:** `RES-03` — Phân tích các kiến trúc phân đoạn được lựa chọn
* **Thuộc đề tài:** Nghiên cứu và ứng dụng kiến trúc U-Net kết hợp cơ chế Attention trong bài toán phân đoạn tổn thương phổi từ ảnh cắt lớp vi tính (CT Scans)
* **Người thực hiện:** Hải (AI & System Engineering)
* **Nguồn gốc:** nội dung nguyên văn của file `Tech_Note_UNet_Architecture.md` (Deliverable của RES-03)
* **Mục đích:** Cung cấp phân tích kỹ thuật chuyên sâu về cấu trúc Encoder–Decoder, Skip Connections, Pooling/Upsampling và Attention Gate nhằm phục vụ trực tiếp thiết kế thực nghiệm mô hình (W3–W4), giải quyết RQ1 và RQ2.

---

## 8. TỔNG QUAN VÀ BỐI CẢNH THỰC NGHIỆM

Trong bài toán phân đoạn tổn thương phổi (kính mờ – GGO, đông đặc – Consolidation) từ ảnh CT lát cắt 2D:
* **Đặc điểm bài toán:** Tổn thương thường chiếm diện tích nhỏ trên lát cắt, ranh giới mờ, độ tương phản thấp so với mô phế nang xung quanh, và có nguy cơ dương tính giả cao ở thành ngực/xương sườn/mạch máu.
* **Bộ 3 mô hình nghiên cứu:**
  1. **Standard U-Net (Baseline đối chuẩn):** Kiến trúc kinh điển với kết nối tắt trực tiếp ghép nối đặc trưng mức thấp và mức cao.
  2. **Attention U-Net (Mô hình nghiên cứu trọng tâm):** Tích hợp module Attention Gate (AG) vào skip connections nhằm lọc nhiễu không gian và làm nổi bật vùng tổn thương.
  3. **SegNet (Mô hình đối chuẩn mở rộng):** Đại diện cho cơ chế lưu vết vị trí cực đại (Pooling Indices) để khôi phục biên không gian mà không truyền trực tiếp toàn bộ feature maps.

---

## 9. PHÂN TÍCH CHI TIẾT KIẾN TRÚC MÔ HÌNH

### 9.1. Standard U-Net (Ronneberger et al., 2015)
* **Encoder (Contracting Path):**
  * Gồm 4 tầng giảm mẫu; mỗi tầng là một khối tích chập kép (`DoubleConv`: $3\times3\text{ Conv} \to \text{BatchNorm} \to \text{ReLU} \to 3\times3\text{ Conv} \to \text{BatchNorm} \to \text{ReLU}$).
  * Sau mỗi tầng, không gian ảnh giảm một nửa qua $2\times2\text{ Max-Pooling}$ (stride = 2), số kênh tăng dần: $1 \to 64 \to 128 \to 256 \to 512$.
  * *Vai trò:* Trích xuất ngữ nghĩa toàn cục và bối cảnh bệnh học ("Đây là dạng tổn thương gì?").
* **Bottleneck:**
  * Đáy chữ U nén không gian xuống $16\times16$ (với đầu vào $256\times256$), số kênh đạt mức tối đa ($1024$). Chứa thông tin trừu tượng mức cao nhất.
* **Decoder (Expanding Path):**
  * Gồm 4 tầng tăng mẫu tương ứng; mỗi bước sử dụng $2\times2\text{ Transposed Convolution}$ (Up-conv / ConvTranspose2d, stride = 2) để nhân đôi kích thước không gian và giảm nửa số kênh.
  * *Skip Connection:* Sao chép trực tiếp feature map ở cùng cấp độ từ Encoder sang Decoder và thực hiện phép ghép nối (`Concatenation`) dọc theo trục channels trước khi đưa qua `DoubleConv`.
  * *Tầng phân lớp cuối:* Lớp tích chập $1\times1\text{ Conv}$ ánh xạ số kênh về 1, kết hợp hàm kích hoạt `Sigmoid` đưa xác suất về khoảng $[0.0, 1.0]$.
* **Ưu điểm & Nhược điểm đối với ảnh CT:**
  * *Ưu điểm:* Khôi phục chi tiết biên tốt nhờ truyền trực tiếp đặc trưng độ phân giải cao qua Skip Connections.
  * *Nhược điểm:* Truyền toàn bộ đặc trưng bao gồm cả mô lành, thành ngực, xương sườn gây nhiễu và dễ tạo ra dương tính giả (False Positives).

---

### 9.2. Attention U-Net (Oktay et al., 2018)
* **Cơ chế tổng thể:**
  * Giữ nguyên cấu trúc đối xứng Encoder, Bottleneck và Decoder của Standard U-Net.
  * Cải tiến then chốt: Đặt thêm module **Attention Gate (AG)** trên từng đường truyền skip connection trước khi thực hiện phép ghép nối (`Concatenation`).
* **Đặc tả toán học Attention Gate (AG):**
  * AG nhận 2 tín hiệu đầu vào:
    1. $x^l$: Bản đồ đặc trưng không gian (Spatial Features) từ tầng $l$ của nhánh Encoder.
    2. $g$: Tín hiệu điều khiển (Gating Signal) lấy từ tầng sâu hơn $l+1$ của Decoder (mang ngữ nghĩa bao quát).
  * Công thức tính hệ số chú ý $\alpha$:
    $$q_{att}^l = \psi^T \left( \sigma_1 \left( W_x^T x^l + W_g^T g + b_g \right) \right) + b_\psi$$
    $$\alpha^l = \sigma_2 \left( q_{att}^l \right)$$
    * Trong đó: $W_x, W_g, \psi$ được cài đặt bằng các phép tích chập $1\times1$; $\sigma_1$ là hàm kích hoạt $\text{ReLU}$; $\sigma_2$ là hàm $\text{Sigmoid}$ để đưa trọng số về dải $[0, 1]$.
  * Đầu ra của Attention Gate:
    $$\hat{x}^l = \alpha^l \odot x^l$$
    * Với $\odot$ là phép nhân từng phần tử (element-wise multiplication).
* **Vai trò trong phân đoạn CT phổi (Phục vụ RQ1):**
  * Tín hiệu gating $g$ hướng dẫn việc tái cân bằng trọng số không gian: triệt tiêu (làm mờ về 0) các vùng mô lành, phế nang bình thường và xương sườn.
  * Làm sáng rực các vùng tổn thương thực sự (đặc biệt là tổn thương dạng kính mờ GGO và các nốt u nhỏ), giúp Decoder tập trung tối đa vào vùng bệnh.

---

### 9.3. SegNet (Badrinarayanan et al., 2017)
* **Cấu trúc tổng thể:**
  * Kiến trúc Encoder–Decoder đối xứng dựa trên cấu trúc các khối VGG-16.
* **Cơ chế Max-Pooling Indices & Max-Unpooling:**
  * *Tại Encoder:* Khi thực hiện phép $2\times2\text{ Max-Pooling}$, SegNet lưu lại ma trận chỉ số vị trí của giá trị lớn nhất trong từng cửa sổ $2\times2$ (gọi là `pooling indices` / `argmax`).
  * *Tại Decoder:* SegNet **không dùng Skip Connections** dạng truyền toàn bộ tensor đặc trưng, cũng không dùng Transposed Convolution có trọng số học được. Thay vào đó, nó sử dụng phép **Max-Unpooling**: đưa các giá trị đặc trưng ở tầng dưới lên đúng vị trí tọa độ đã lưu trong `pooling indices`, các vị trí còn lại trong cửa sổ $2\times2$ được điền giá trị 0.
  * Sau đó, feature map thưa thớt này được đưa qua các lớp tích chập chuẩn để làm mịn và phục hồi thông tin.
* **Ưu điểm & Nhược điểm đối với ảnh CT (Phục vụ RQ2):**
  * *Ưu điểm:* Tiết kiệm bộ nhớ đáng kể (chỉ lưu ma trận chỉ số nguyên thay vì lưu toàn bộ tensor số thực 32-bit), duy trì cấu trúc biên của các vật thể nhỏ gọn.
  * *Nhược điểm:* Việc thiếu đường truyền đặc trưng trực tiếp khiến thông tin ngữ nghĩa mức thấp dễ bị thất thoát; vùng dự đoán bên trong tổn thương có thể bị rỗng hoặc không đồng nhất.

---

## 10. BẢNG SO SÁNH ĐỐI CHUẨN KỸ THUẬT PHỤC VỤ THỰC NGHIỆM

| Tiêu chí kỹ thuật | Standard U-Net | Attention U-Net | SegNet |
| :--- | :--- | :--- | :--- |
| **Vai trò nghiên cứu** | Baseline chính | Mô hình nghiên cứu trọng tâm | Baseline đối chứng bổ sung |
| **Cơ chế Downsampling** | $2\times2\text{ Max-Pooling}$ | $2\times2\text{ Max-Pooling}$ | $2\times2\text{ Max-Pooling}$ (lưu `indices`) |
| **Cơ chế Upsampling** | Transposed Conv ($2\times2$, stride 2) | Transposed Conv ($2\times2$, stride 2) | Max-Unpooling (dựa trên `indices`) |
| **Cơ chế liên kết (Skip)** | Ghép nối trực tiếp (`Concat`) | Lọc qua Attention Gate rồi `Concat` | Không có (chỉ truyền `pooling indices`) |
| **Số lượng tham số (Params)** | Trung bình (~31.0M) | Tăng nhẹ (+1.5% đến 2.0% do AG) | Thấp đến trung bình (~29.4M) |
| **Bộ nhớ VRAM (Training)** | Trung bình | Tăng nhẹ (thêm tensor gating) | Tiết kiệm nhất trong 3 mô hình |
| **Độ nhạy biên tổn thương** | Tốt, nhưng dễ nhiễu mô lành | Tốt nhất, lọc nhiễu nền | Tốt ở biên nốt nhỏ, kém ở vùng mờ |
| **Phù hợp với dạng tổn thương** | Tổn thương trung bình & lớn | Tổn thương nhỏ, kính mờ (GGO) | Nốt đơn độc có ranh giới rõ |

---

## 11. ĐỊNH HƯỚNG THIẾT KẾ THỰC NGHIỆM ĐỒNG BỘ (W3 – W5)

Để đảm bảo kết quả so sánh khách quan và đáp ứng các quy chuẩn khoa học (Evidence-based):

1. **Chuẩn hóa giao diện Input/Output:**
   * Cả 3 mô hình nhận chung đầu vào Tensor: `[Batch_Size, 1, 256, 256]`, chuẩn hóa Min-Max $[0.0, 1.0]$ sau khi cắt ngưỡng HU $[-1000, 400]$.
   * Cả 3 mô hình trả về ma trận xác suất nhị phân: `[Batch_Size, 1, 256, 256]` qua hàm `Sigmoid`.

2. **Đồng nhất hàm tổn thất (Hybrid Loss):**
   * Sử dụng thống nhất:
     $$\mathcal{L}_{\text{hybrid}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{SoftDice}}$$
   * Soft Dice Loss bắt buộc có hệ số làm mịn $\epsilon = 10^{-5}$ để tránh lỗi chia cho 0.

3. **Cấu hình siêu tham số (Hyperparameters):**
   * Optimizer: `AdamW` (learning rate ban đầu $10^{-4}$, weight decay $10^{-4}$).
   * Scheduler: `CosineAnnealingLR`.
   * Batch size: 16 (phù hợp VRAM máy trạm), tối đa 80–100 epochs kèm Early Stopping (patience = 15 trên Validation Set).
   * Cùng hạt giống ngẫu nhiên (random seed) cho cả 3 lượt huấn luyện.

4. **Trọng tâm kiểm chứng câu hỏi nghiên cứu:**
   * **Kiểm chứng RQ1 (Attention U-Net vs Standard U-Net):**
     * Đánh giá độ chênh lệch chỉ số DSC và IoU, đặc biệt trên tập con các tổn thương nhỏ (diện tích $< 5\%$ diện tích phổi) và các ca bệnh có tổn thương kính mờ (GGO).
     * Kiểm định ý nghĩa thống kê bằng paired Wilcoxon signed-rank test ($p\text{-value} < 0.05$).
   * **Kiểm chứng RQ2 (U-Net vs SegNet):**
     * So sánh khả năng duy trì độ sắc nét của đường viền biên (Boundary Delineation) qua các phép phóng to (Zoom-in visualization).
     * Đánh giá sự đánh đổi (Trade-off) giữa độ chính xác phân đoạn, số lượng tham số (FLOPs/Params) và độ trễ suy luận (latency ms/slice) phục vụ triển khai Web Demo.
