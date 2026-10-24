# 02 — WBS & ROADMAP

> **Mục đích:** Chia toàn bộ đề tài nghiên cứu thành các **phase, task và sub-task cụ thể**, đồng thời xác định rõ **ai làm, làm gì, làm khi nào, phối hợp với ai và phải bàn giao kết quả gì**.
>
> **Thời gian:** 8 tuần
> **Số thành viên:** 5
> **Nguyên tắc:** Tất cả thành viên đều có vai trò và trách nhiệm ngang nhau. Không sử dụng khái niệm Leader/Lead trong phân công công việc.

---

# 1. TEAM

| Thành viên | Phụ trách chính         | Trọng tâm                                                      |
| ---------- | ----------------------- | -------------------------------------------------------------- |
| **Duy**    | Research & Paper        | Nghiên cứu, RQ, tài liệu, Paper, tổng hợp                      |
| **Hải**    | AI & System Engineering | Thuật toán, mô hình, training, kỹ thuật hệ thống, Web Demo     |
| **Đông**   | Data & Evaluation       | Dataset, preprocessing, DataLoader, metrics, đánh giá          |
| **Hậu**    | Baseline & Experiment   | SegNet, baseline experiment, visualization, hỗ trợ thực nghiệm |
| **Hào**    | QA & Reproducibility    | Kiểm thử, failure cases, Evidence, reproducibility, audit      |

### Nguyên tắc phân công

```text
Duy    → Nghiên cứu & Paper
Hải    → Thuật toán & Xây dựng sản phẩm
Đông   → Dữ liệu & Đánh giá
Hậu    → Baseline & Thực nghiệm
Hào    → QA & Reproducibility
```

> **Hải phụ trách chính phần kỹ thuật và xây dựng sản phẩm**, đặc biệt là các phần liên quan đến thuật toán, kiến trúc mô hình, training pipeline, tích hợp model và triển khai Web Demo.
>
> Tuy nhiên, Hải vẫn là **một thành viên của nhóm**, không phải cấp quản lý của các thành viên khác.

---

# 2. PROJECT ROADMAP

Toàn bộ dự án được chia thành 8 giai đoạn:

```text
W1
Research Foundation
       ↓
W2
Dataset & Data Pipeline
       ↓
W3
Baseline Models
       ↓
W4
Attention U-Net
       ↓
W5
Experiment & Evaluation
       ↓
W6
Web Demo
       ↓
W7
Scientific Paper
       ↓
W8
Final Report & Defense
```

---

# 3. PHÂN BỔ CÔNG VIỆC TỔNG QUÁT

| Phase                  | Duy | Hải | Đông | Hậu | Hào |
| ---------------------- | :-: | :-: | :--: | :-: | :-: |
| **W1 Research**        |  ●  |  ○  |  ○   |  ○  |  ○  |
| **W2 Dataset**         |  ○  |  ○  |  ●   |  ○  |  ○  |
| **W3 Baseline**        |  ○  |  ●  |  ○   |  ●  |  ○  |
| **W4 Attention U-Net** |  ○  |  ●  |  ○   |  ○  |  ○  |
| **W5 Evaluation**      |  ●  |  ○  |  ●   |  ○  |  ●  |
| **W6 Web Demo**        |  ○  |  ●  |  ○   |  ○  |  ●  |
| **W7 Paper**           |  ●  |  ○  |  ●   |  ○  |  ●  |
| **W8 Finalization**    |  ●  |  ●  |  ○   |  ○  |  ●  |

### Ký hiệu

- **●** = Phụ trách chính
- **○** = Phối hợp / hỗ trợ

---

# 4. MASTER TASK LIST

| ID         | Task                         | Phụ trách  | Tuần | Deliverable            |
| ---------- | ---------------------------- | ---------- | ---- | ---------------------- |
| `RES-01`   | Literature Review            | Duy        | W1   | Literature notes       |
| `RES-02`   | Research Questions           | Duy        | W1   | RQ1, RQ2               |
| `RES-03`   | Research Methodology Outline | Duy        | W1   | Methodology outline    |
| `DATA-01`  | Dataset Investigation        | Đông       | W1   | Dataset decision       |
| `DATA-02`  | Dataset Acquisition          | Đông       | W2   | Raw dataset            |
| `DATA-03`  | CT Preprocessing             | Đông       | W2   | Preprocessing pipeline |
| `DATA-04`  | Mask Processing              | Đông       | W2   | Binary masks           |
| `DATA-05`  | Patient-level Split          | Đông       | W2   | Train/Val/Test split   |
| `DATA-06`  | DataLoader                   | Đông       | W2   | PyTorch DataLoader     |
| `MODEL-01` | U-Net                        | Hải        | W3   | U-Net source           |
| `MODEL-02` | Training Pipeline            | Hải        | W3   | Training framework     |
| `MODEL-03` | SegNet                       | Hậu        | W3   | SegNet source          |
| `EXP-01`   | U-Net Experiment             | Hải        | W3   | Experiment result      |
| `EXP-02`   | SegNet Experiment            | Hậu        | W3   | Experiment result      |
| `MODEL-04` | Attention Gate               | Hải        | W4   | Attention module       |
| `MODEL-05` | Attention U-Net              | Hải        | W4   | Attention U-Net        |
| `EXP-03`   | Attention U-Net Experiment   | Hải        | W4   | Experiment result      |
| `EVAL-01`  | Evaluation Pipeline          | Đông       | W5   | Metrics pipeline       |
| `EVAL-02`  | Model Comparison             | Đông       | W5   | Comparison table       |
| `EVAL-03`  | Visualization                | Hậu        | W5   | Result figures         |
| `EVAL-04`  | Failure-case Analysis        | Hào        | W5   | Failure-case report    |
| `EVAL-05`  | RQ1 / RQ2 Analysis           | Duy        | W5   | Research findings      |
| `DEMO-01`  | Inference Pipeline           | Hải        | W6   | Inference module       |
| `DEMO-02`  | Web Interface                | Hải        | W6   | Web Demo               |
| `DEMO-03`  | Demo Testing                 | Hào        | W6   | Test report            |
| `PAPER-01` | Introduction                 | Duy        | W7   | Paper section          |
| `PAPER-02` | Methodology                  | Duy + Hải  | W7   | Paper section          |
| `PAPER-03` | Experimental Setup           | Hải + Đông | W7   | Paper section          |
| `PAPER-04` | Results                      | Duy + Đông | W7   | Tables/Figures         |
| `PAPER-05` | Discussion                   | Duy        | W7   | Paper section          |
| `PAPER-06` | References                   | Duy        | W7   | IEEE References        |
| `FINAL-01` | Final Report                 | Duy        | W8   | Final report           |
| `FINAL-02` | Presentation Slides          | Duy + All  | W8   | Slides                 |
| `FINAL-03` | Demo Rehearsal               | Hải + Hào  | W8   | Rehearsal              |
| `FINAL-04` | Defense Q&A                  | All        | W8   | Q&A Bank               |
| `QA-01`    | Reproducibility Audit        | Hào        | W8   | QA report              |
| `QA-02`    | Repository Audit             | Hào        | W8   | Clean repository       |
| `QA-03`    | Final Research Review        | Duy + Hào  | W8   | Final checklist        |

---

# 5. PHÂN CÔNG CHI TIẾT — DUY

## Vai trò

**Research & Paper**

## Trách nhiệm chính

- Tìm kiếm và tổng hợp tài liệu.
- Xây dựng Research Questions.
- Xác định hướng nghiên cứu.
- Tổng hợp kết quả từ các thành viên.
- Viết và hoàn thiện Paper.
- Chuẩn bị Report và Presentation.

### Task của Duy

| ID         | Công việc                    | Tuần  | Kết quả            |
| ---------- | ---------------------------- | ----- | ------------------ |
| `RES-01`   | Tìm kiếm tài liệu liên quan  | W1    | ≥ 20 tài liệu      |
| `RES-02`   | Xác định RQ1, RQ2            | W1    | Research Questions |
| `RES-03`   | Xây dựng Methodology Outline | W1    | Outline            |
| `RES-04`   | Quản lý tài liệu Zotero      | W1–W7 | IEEE References    |
| `EVAL-05`  | Phân tích RQ1/RQ2            | W5    | Research findings  |
| `PAPER-01` | Viết Introduction            | W7    | Introduction       |
| `PAPER-02` | Viết Methodology             | W7    | Methodology        |
| `PAPER-04` | Tổng hợp Results             | W7    | Results            |
| `PAPER-05` | Viết Discussion              | W7    | Discussion         |
| `PAPER-06` | References                   | W7    | References         |
| `FINAL-01` | Hoàn thiện Final Report      | W8    | Final Report       |
| `FINAL-02` | Tổng hợp Presentation        | W8    | Slides             |
| `QA-03`    | Final Research Review        | W8    | Final checklist    |

### Duy cần nhận dữ liệu từ

```text
Đông → Dataset / Metrics / Evaluation
Hải  → Model / Training / Technical details
Hậu  → Baseline / Visualization
Hào  → QA / Failure cases / Evidence
```

---

# 6. PHÂN CÔNG CHI TIẾT — HẢI

## Vai trò

**AI & System Engineering**

## Trách nhiệm chính

Hải là người phụ trách chính các phần:

- Thuật toán.
- Kiến trúc Deep Learning.
- U-Net.
- Attention U-Net.
- Training pipeline.
- Model integration.
- Inference.
- Web Demo.
- Các vấn đề kỹ thuật quan trọng của sản phẩm.

### Task của Hải

| ID         | Công việc                   | Tuần  | Kết quả            |
| ---------- | --------------------------- | ----- | ------------------ |
| `TECH-01`  | Thiết lập môi trường AI/GPU | W1    | Environment        |
| `MODEL-01` | Xây dựng U-Net              | W3    | U-Net source       |
| `MODEL-02` | Xây dựng Training Pipeline  | W3    | Training framework |
| `EXP-01`   | Train U-Net                 | W3    | U-Net checkpoint   |
| `MODEL-04` | Xây dựng Attention Gate     | W4    | Attention module   |
| `MODEL-05` | Xây dựng Attention U-Net    | W4    | Attention U-Net    |
| `EXP-03`   | Train Attention U-Net       | W4    | Checkpoint         |
| `TECH-02`  | Tối ưu / debug training     | W3–W5 | Technical notes    |
| `DEMO-01`  | Xây dựng Inference Pipeline | W6    | Inference          |
| `DEMO-02`  | Xây dựng Web Demo           | W6    | Streamlit/Gradio   |
| `PAPER-02` | Cung cấp technical details  | W7    | Methodology        |
| `PAPER-03` | Experimental setup          | W7    | Technical setup    |
| `FINAL-03` | Demo rehearsal              | W8    | Working demo       |

### Chuỗi công việc chính của Hải

```text
U-Net
  ↓
Training Pipeline
  ↓
Baseline Experiment
  ↓
Attention Gate
  ↓
Attention U-Net
  ↓
Best Checkpoint
  ↓
Inference
  ↓
Web Demo
```

### Hải bàn giao

- Source code.
- Model architecture.
- Training configuration.
- Checkpoint.
- Experiment log.
- Inference pipeline.
- Web Demo.
- Technical explanation cho Paper.

---

# 7. PHÂN CÔNG CHI TIẾT — ĐÔNG

## Vai trò

**Data & Evaluation**

## Trách nhiệm chính

- Dataset.
- Preprocessing.
- Mask.
- Data split.
- DataLoader.
- Metrics.
- Evaluation.
- Model comparison.

### Task của Đông

| ID         | Công việc           | Tuần | Kết quả          |
| ---------- | ------------------- | ---- | ---------------- |
| `DATA-01`  | Khảo sát dataset    | W1   | Dataset decision |
| `DATA-02`  | Thu thập dataset    | W2   | Raw dataset      |
| `DATA-03`  | CT preprocessing    | W2   | Pipeline         |
| `DATA-04`  | Xử lý binary masks  | W2   | Masks            |
| `DATA-05`  | Patient-level split | W2   | Train/Val/Test   |
| `DATA-06`  | Xây dựng DataLoader | W2   | DataLoader       |
| `EVAL-01`  | Evaluation pipeline | W5   | Metrics          |
| `EVAL-02`  | Model comparison    | W5   | Comparison table |
| `PAPER-03` | Experimental Setup  | W7   | Setup            |
| `PAPER-04` | Results             | W7   | Results table    |

### Metrics

Đông chịu trách nhiệm tính toán:

- Dice / DSC
- IoU / Jaccard
- Precision
- Recall

---

# 8. PHÂN CÔNG CHI TIẾT — HẬU

## Vai trò

**Baseline & Experiment**

## Trách nhiệm chính

- SegNet.
- Baseline comparison.
- Visualization.
- Hỗ trợ thực nghiệm.
- Hỗ trợ kiểm tra kết quả model.

### Task của Hậu

| ID         | Công việc                 | Tuần  | Kết quả            |
| ---------- | ------------------------- | ----- | ------------------ |
| `MODEL-03` | Xây dựng SegNet           | W3    | SegNet source      |
| `EXP-02`   | Train SegNet              | W3    | SegNet checkpoint  |
| `EXP-04`   | Kiểm tra prediction       | W3–W4 | Prediction samples |
| `EVAL-03`  | Visualization             | W5    | Figures            |
| `EVAL-06`  | Chuẩn bị GT vs Prediction | W5    | Comparison images  |
| `TECH-03`  | Hỗ trợ debugging          | W3–W5 | Technical notes    |
| `PAPER-04` | Cung cấp figures          | W7    | Paper figures      |
| `FINAL-02` | Hỗ trợ presentation       | W8    | Slides             |

### Chuỗi công việc

```text
SegNet
  ↓
Training
  ↓
Baseline Experiment
  ↓
Prediction
  ↓
Visualization
  ↓
Comparison
```

---

# 9. PHÂN CÔNG CHI TIẾT — HÀO

## Vai trò

**QA & Reproducibility**

## Trách nhiệm chính

- Kiểm tra chất lượng.
- Theo dõi Evidence.
- Failure-case analysis.
- Kiểm tra tính reproducible.
- Test Web Demo.
- Audit repository.
- Kiểm tra tính nhất quán giữa code, experiment và Paper.

### Task của Hào

| ID         | Công việc                   | Tuần  | Kết quả              |
| ---------- | --------------------------- | ----- | -------------------- |
| `QA-01`    | Thiết lập QA checklist      | W1    | QA checklist         |
| `QA-02`    | Kiểm tra dataset pipeline   | W2    | QA notes             |
| `QA-03`    | Kiểm tra experiment records | W3–W4 | QA notes             |
| `EVAL-04`  | Failure-case analysis       | W5    | Failure report       |
| `DEMO-03`  | Test Web Demo               | W6    | Test report          |
| `PAPER-QA` | Kiểm tra Experimental Setup | W7    | QA feedback          |
| `QA-04`    | Kiểm tra Evidence           | W7    | Evidence checklist   |
| `QA-05`    | Reproducibility Audit       | W8    | QA report            |
| `QA-06`    | Repository Audit            | W8    | Repository checklist |
| `QA-03`    | Final Research Review       | W8    | Final checklist      |

### Hào kiểm tra đặc biệt

```text
Code
 ↓
Experiment
 ↓
Evidence
 ↓
Result
 ↓
Paper
```

Phải bảo đảm 5 tầng này **khớp nhau**.

---

# 10. ROADMAP CHI TIẾT 8 TUẦN

---

## W1 — RESEARCH FOUNDATION

### Mục tiêu

Xác định rõ:

- Nghiên cứu vấn đề gì?
- Dataset nào?
- Mô hình nào?
- RQ1/RQ2 là gì?
- Thực nghiệm sẽ được tiến hành như thế nào?

### Phân công

| Người    | Task                                            |
| -------- | ----------------------------------------------- |
| **Duy**  | Literature Review, RQ, Research Outline, Zotero |
| **Hải**  | Setup Python/PyTorch/GPU, chuẩn bị môi trường   |
| **Đông** | Khảo sát Dataset                                |
| **Hậu**  | Khảo sát U-Net / SegNet implementation          |
| **Hào**  | QA checklist + Evidence rules                   |

### Deliverables

- [ ] Literature collection
- [ ] RQ1
- [ ] RQ2
- [ ] Dataset decision
- [ ] Development environment
- [ ] QA checklist

---

# W2 — DATASET & DATA PIPELINE

### Mục tiêu

Hoàn thành pipeline dữ liệu.

### Phân công

| Người    | Task                                                |
| -------- | --------------------------------------------------- |
| **Duy**  | Theo dõi phạm vi nghiên cứu                         |
| **Hải**  | Hỗ trợ tích hợp DataLoader với training             |
| **Đông** | Dataset + preprocessing + mask + split + DataLoader |
| **Hậu**  | Kiểm tra sample data và visualization               |
| **Hào**  | QA dataset pipeline                                 |

### Deliverables

```text
Raw Dataset
    ↓
Preprocessing
    ↓
Mask
    ↓
Patient-level Split
    ↓
DataLoader
```

- [ ] Dataset sạch
- [ ] HU preprocessing
- [ ] Binary masks
- [ ] Train/Val/Test
- [ ] DataLoader
- [ ] Sample visualization

---

# W3 — BASELINE MODELS

### Mục tiêu

Xây dựng baseline:

- U-Net
- SegNet

### Phân công

| Người    | Task                                         |
| -------- | -------------------------------------------- |
| **Duy**  | Theo dõi kết quả + cập nhật Research Notes   |
| **Hải**  | U-Net + Training Pipeline + U-Net Experiment |
| **Đông** | Hỗ trợ DataLoader + metrics                  |
| **Hậu**  | SegNet + SegNet Experiment                   |
| **Hào**  | Kiểm tra experiment records                  |

### Deliverables

- [ ] U-Net
- [ ] SegNet
- [ ] Training Pipeline
- [ ] U-Net checkpoint
- [ ] SegNet checkpoint
- [ ] Experiment IDs
- [ ] Training logs

---

# W4 — ATTENTION U-NET

### Mục tiêu

Xây dựng và huấn luyện Attention U-Net.

### Phân công

| Người    | Task                                        |
| -------- | ------------------------------------------- |
| **Duy**  | Theo dõi RQ1                                |
| **Hải**  | Attention Gate + Attention U-Net + training |
| **Đông** | Chuẩn bị evaluation                         |
| **Hậu**  | Hỗ trợ prediction/visualization             |
| **Hào**  | Kiểm tra reproducibility                    |

### Deliverables

- [ ] Attention Gate
- [ ] Attention U-Net
- [ ] Training
- [ ] Checkpoint
- [ ] Training log
- [ ] Experiment ID

---

# W5 — EXPERIMENT & EVALUATION

### Mục tiêu

So sánh 3 mô hình:

```text
U-Net
   vs
SegNet
   vs
Attention U-Net
```

### Phân công

| Người    | Task                                  |
| -------- | ------------------------------------- |
| **Duy**  | Phân tích RQ1/RQ2                     |
| **Hải**  | Technical analysis + hỗ trợ inference |
| **Đông** | Metrics + Model Comparison            |
| **Hậu**  | Visualization                         |
| **Hào**  | Failure Cases                         |

### Deliverables

- [ ] Dice
- [ ] IoU
- [ ] Precision
- [ ] Recall
- [ ] Comparison table
- [ ] GT vs Prediction
- [ ] Failure cases
- [ ] RQ1 analysis
- [ ] RQ2 analysis

---

# W6 — WEB DEMO

### Mục tiêu

Đóng gói model thành Web Demo.

### Phân công

| Người    | Task                                     |
| -------- | ---------------------------------------- |
| **Duy**  | Kiểm tra demo phù hợp phạm vi nghiên cứu |
| **Hải**  | Inference + Web Demo + Model Integration |
| **Đông** | Kiểm tra preprocessing consistency       |
| **Hậu**  | Hỗ trợ UI/result visualization           |
| **Hào**  | Integration Test                         |

### Flow

```text
Upload CT
    ↓
Preprocessing
    ↓
Model
    ↓
Prediction
    ↓
Segmentation Mask
    ↓
Visualization
```

### Deliverables

- [ ] Inference pipeline
- [ ] Web interface
- [ ] Upload image
- [ ] Model inference
- [ ] Segmentation visualization
- [ ] Demo test

---

# W7 — SCIENTIFIC PAPER

### Mục tiêu

Hoàn thành bản thảo Paper.

### Phân công

| Người    | Task                                                 |
| -------- | ---------------------------------------------------- |
| **Duy**  | Introduction + Methodology + Discussion + References |
| **Hải**  | Technical Methodology + Experimental Setup           |
| **Đông** | Dataset + Metrics + Results                          |
| **Hậu**  | Figures + Baseline results                           |
| **Hào**  | Paper QA + Evidence verification                     |

### Cấu trúc

```text
1. Introduction
2. Related Work
3. Methodology
4. Experimental Setup
5. Results
6. Discussion
7. Conclusion
8. References
```

---

# W8 — FINALIZATION & DEFENSE

### Mục tiêu

Đóng gói toàn bộ sản phẩm nghiên cứu.

| Người    | Task                                    |
| -------- | --------------------------------------- |
| **Duy**  | Final Report + tổng hợp Slides          |
| **Hải**  | Technical Demo + Technical Q&A          |
| **Đông** | Kiểm tra Results / Metrics              |
| **Hậu**  | Chuẩn bị Figures / Presentation         |
| **Hào**  | QA + Repository Audit + Reproducibility |

### Deliverables

- [ ] Final Report
- [ ] Presentation Slides
- [ ] Working Web Demo
- [ ] Q&A Bank
- [ ] Repository
- [ ] Reproducibility Audit
- [ ] Final QA

---

# 11. DEPENDENCY GIỮA CÁC THÀNH VIÊN

## Luồng Dataset → Model

```text
Đông
Dataset
  ↓
Preprocessing
  ↓
DataLoader
  ↓
Hải ─────────────── Hậu
U-Net              SegNet
  │                  │
  └────────┬─────────┘
           ↓
        Training
           ↓
        Results
```

## Luồng Model → Evaluation

```text
Hải
U-Net / Attention U-Net
        │
        ├─────────────┐
        ↓             ↓
      Đông           Hào
   Evaluation      QA / Failure
        │             │
        └──────┬──────┘
               ↓
              Duy
           RQ Analysis
```

## Luồng Evaluation → Paper

```text
Đông ── Metrics ───────┐
Hậu  ── Figures ───────┤
Hào  ── QA/Evidence ───┤
Hải  ── Technical ─────┤
                       ↓
                      Duy
                       ↓
                     Paper
```

---

# 12. BẢNG BÀN GIAO

| Người bàn giao | →   | Người nhận | Nội dung                   |
| -------------- | --- | ---------- | -------------------------- |
| Đông           | →   | Hải        | Dataset + DataLoader       |
| Đông           | →   | Hậu        | Dataset + sample data      |
| Hải            | →   | Đông       | Prediction + checkpoint    |
| Hậu            | →   | Đông       | SegNet prediction          |
| Đông           | →   | Duy        | Metrics + comparison       |
| Hậu            | →   | Duy        | Figures + baseline results |
| Hải            | →   | Duy        | Technical details          |
| Hào            | →   | Duy        | QA + failure cases         |
| Hải            | →   | Hào        | Demo để test               |
| Tất cả         | →   | Duy        | Nội dung phục vụ Paper     |

---

# 13. DEFINITION OF DONE

Một task chỉ được xem là **DONE** khi:

- [ ] Công việc đã thực hiện.
- [ ] Output đã được tạo.
- [ ] Output đã được kiểm tra.
- [ ] Code / file đã được lưu đúng nơi.
- [ ] Evidence đã được lưu.
- [ ] Thành viên phụ trách xác nhận hoàn thành.

### Đối với Experiment

Bắt buộc có:

```text
Experiment ID
+
Dataset / Split
+
Model
+
Configuration
+
Training Log
+
Checkpoint
+
Metrics
+
Result
```

> **Không có Experiment ID → Không sử dụng kết quả làm kết quả chính thức trong Paper.**

---

# 14. TASK STATUS

Mỗi task sử dụng một trong 6 trạng thái:

| Status        | Ý nghĩa                  |
| ------------- | ------------------------ |
| `TODO`        | Chưa thực hiện           |
| `IN PROGRESS` | Đang thực hiện           |
| `BLOCKED`     | Đang bị chặn             |
| `REVIEW`      | Đã làm, đang kiểm tra    |
| `DONE`        | Hoàn thành + có Evidence |
| `CANCELLED`   | Đã hủy                   |

---

# 15. WEEKLY CHECK-IN

Cuối mỗi tuần, mỗi thành viên cập nhật:

```md
## Weekly Update — [Tên]

### Completed

- [Task ID] ...
- [Task ID] ...

### Evidence

- ...

### Blocked

- ...

### Next Week

- ...

### Need Support

- ...
```

### 5 câu hỏi bắt buộc

1. Tuần này đã làm gì?
2. Đã hoàn thành task nào?
3. Evidence nằm ở đâu?
4. Đang bị block ở đâu?
5. Tuần sau cần làm gì?

---

# 16. CRITICAL PATH

Các công việc ảnh hưởng trực tiếp đến tiến độ toàn dự án:

```text
Dataset
  ↓
Preprocessing
  ↓
Patient-level Split
  ↓
DataLoader
  ↓
U-Net / SegNet
  ↓
Attention U-Net
  ↓
Experiments
  ↓
Evaluation
  ↓
RQ Analysis
  ↓
Paper
  ↓
Final Report
```

### Task cần ưu tiên không để trễ

- `DATA-02`
- `DATA-03`
- `DATA-05`
- `DATA-06`
- `MODEL-01`
- `MODEL-02`
- `MODEL-04`
- `MODEL-05`
- `EXP-01`
- `EXP-02`
- `EXP-03`
- `EVAL-01`
- `EVAL-02`
- `EVAL-05`
- `PAPER-04`
- `FINAL-01`

---

# 17. QUY TẮC KHI TASK BỊ BLOCKED

Khi gặp vấn đề:

```text
Task
 ↓
Xác định nguyên nhân
 ↓
Xác định task / người đang phụ thuộc
 ↓
Thông báo
 ↓
Hỗ trợ xử lý
 ↓
Cập nhật Status
```

Ví dụ:

```text
MODEL-01
U-Net
   ↓
BLOCKED

Reason:
DATA-06 DataLoader chưa hoàn thành.

Action:
Đông hoàn thiện DataLoader.
Hải kiểm tra integration.
```

Không tự ý thay đổi dependency lớn nếu chưa thống nhất trong nhóm.

---

# 18. QUY TẮC THAY ĐỔI PHÂN CÔNG

Trong quá trình thực hiện, nếu cần thay đổi người phụ trách:

```text
Task hiện tại
     ↓
Lý do thay đổi
     ↓
Người nhận task mới
     ↓
Kiểm tra ảnh hưởng
     ↓
Cập nhật tài liệu
```

Việc thay đổi phân công **không mang ý nghĩa cấp bậc**.

Mục tiêu chỉ là bảo đảm:

> **Đúng người — đúng việc — đúng thời điểm.**

---

# 19. FINAL TASK MATRIX

| Thành viên | Research | Data | U-Net | Attention | SegNet | Evaluation | Demo | Paper | QA  |
| ---------- | :------: | :--: | :---: | :-------: | :----: | :--------: | :--: | :---: | :-: |
| **Duy**    |    ●     |  ○   |   ○   |     ○     |   ○    |     ●      |  ○   |   ●   |  ○  |
| **Hải**    |    ○     |  ○   |   ●   |     ●     |   ○    |     ○      |  ●   |   ○   |  ○  |
| **Đông**   |    ○     |  ●   |   ○   |     ○     |   ○    |     ●      |  ○   |   ●   |  ○  |
| **Hậu**    |    ○     |  ○   |   ○   |     ○     |   ●    |     ●      |  ○   |   ○   |  ○  |
| **Hào**    |    ○     |  ○   |   ○   |     ○     |   ○    |     ●      |  ●   |   ○   |  ●  |

---

# 20. PROJECT COMPLETION CHECKLIST

## Research

- [ ] Literature Review
- [ ] Research Questions
- [ ] Methodology
- [ ] Dataset decision

## Dataset

- [ ] Dataset acquired
- [ ] Preprocessing
- [ ] Binary masks
- [ ] Patient-level split
- [ ] DataLoader

## Models

- [ ] U-Net
- [ ] SegNet
- [ ] Attention Gate
- [ ] Attention U-Net

## Experiments

- [ ] U-Net experiment
- [ ] SegNet experiment
- [ ] Attention U-Net experiment
- [ ] Checkpoints
- [ ] Experiment IDs
- [ ] Training logs

## Evaluation

- [ ] Dice / DSC
- [ ] IoU / Jaccard
- [ ] Precision
- [ ] Recall
- [ ] Model comparison
- [ ] Visualization
- [ ] Failure cases
- [ ] RQ1
- [ ] RQ2

## Product

- [ ] Inference
- [ ] Web Demo
- [ ] Upload CT
- [ ] Segmentation result
- [ ] Visualization
- [ ] Demo testing

## Paper

- [ ] Introduction
- [ ] Related Work
- [ ] Methodology
- [ ] Experimental Setup
- [ ] Results
- [ ] Discussion
- [ ] Conclusion
- [ ] References

## Final

- [ ] Final Report
- [ ] Slides
- [ ] Demo
- [ ] Q&A
- [ ] Repository Audit
- [ ] Reproducibility Audit
- [ ] Final QA

---

# 21. NGUYÊN TẮC CUỐI CÙNG

> **Mỗi thành viên đều là một thành viên của nhóm. Không có Leader/Lead trong phân công.**

> **Duy phụ trách Research & Paper.**

> **Hải phụ trách chính AI, thuật toán, kỹ thuật hệ thống và xây dựng sản phẩm.**

> **Đông phụ trách Data & Evaluation.**

> **Hậu phụ trách Baseline & Experiment.**

> **Hào phụ trách QA & Reproducibility.**

Và toàn bộ công việc phải được quản lý theo chuỗi:

```text
AI LÀM?
   ↓
LÀM VIỆC GÌ?
   ↓
LÀM TUẦN NÀO?
   ↓
PHỤ THUỘC VÀO AI?
   ↓
BÀN GIAO GÌ?
   ↓
EVIDENCE Ở ĐÂU?
```

> **Không có Evidence → Không Done.**
>
> **Không có Experiment ID → Không đưa kết quả experiment vào Paper.**
>
> **Không rõ người phụ trách → Task chưa được phân công hoàn chỉnh.**
