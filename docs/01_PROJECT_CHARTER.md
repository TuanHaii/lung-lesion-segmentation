# PROJECT CHARTER

## Nghiên cứu và ứng dụng kiến trúc U-Net kết hợp cơ chế Attention trong bài toán phân đoạn tổn thương phổi từ ảnh cắt lớp vi tính

> **Project Type:** Scientific Research Project
> **Research Field:** Computer Science / Artificial Intelligence / Medical Computer Vision
> **Project Duration:** 8 weeks
> **Team Size:** 5 members
> **Document Status:** Baseline
> **Document Owner:** Research Team
> **Primary Coordination:** M1 – Research Coordination & Paper

---

# 1. Project Overview

## 1.1. Tên đề tài

**Tiếng Việt**

> Nghiên cứu và ứng dụng kiến trúc U-Net kết hợp cơ chế Attention trong bài toán phân đoạn tổn thương phổi từ ảnh cắt lớp vi tính (CT Scans) hỗ trợ chẩn đoán y tế.

**Tiếng Anh**

> Deep Learning-based Lung Lesion Segmentation from Computed Tomography Scans Using Attention U-Net for Medical Diagnosis Support.

Đề tài tập trung nghiên cứu bài toán **phân đoạn tổn thương phổi ở mức pixel** từ ảnh CT bằng các kiến trúc Deep Learning, trong đó **Attention U-Net** là kiến trúc nghiên cứu chính và được đối chiếu với các mô hình phân đoạn khác.

Phạm vi nghiên cứu được xây dựng trên dữ liệu CT công khai, tập trung vào việc xây dựng pipeline xử lý dữ liệu, huấn luyện mô hình, đánh giá định lượng và trực quan hóa kết quả. Nghiên cứu không triển khai trên bệnh nhân thực tế và không thay thế quyết định chẩn đoán của bác sĩ.

---

# 2. Research Context

Phân đoạn tổn thương phổi trên ảnh CT là một bài toán quan trọng trong Medical Image Segmentation.

Trong quy trình thủ công, việc xác định vùng tổn thương trên nhiều lát cắt CT có thể tốn nhiều thời gian và phụ thuộc vào người thực hiện. Một hệ thống phân đoạn tự động có khả năng xác định vùng tổn thương ở mức pixel có thể hỗ trợ:

- Khoanh vùng tổn thương.
- Định lượng diện tích hoặc mức độ tổn thương.
- Chuẩn hóa quá trình phân tích hình ảnh.
- Hỗ trợ trực quan hóa kết quả cho người sử dụng.
- Làm cơ sở cho các nghiên cứu tiếp theo về Computer-Aided Diagnosis.

Trong phạm vi đề tài, nhóm tập trung vào việc nghiên cứu khả năng của **Attention U-Net** trong việc cải thiện khả năng tập trung vào vùng tổn thương so với kiến trúc U-Net tiêu chuẩn.

---

# 3. Vision

Xây dựng một pipeline nghiên cứu có khả năng:

```text
CT Scan
   ↓
Preprocessing
   ↓
Normalized CT Image
   ↓
Segmentation Model
   ├── U-Net
   ├── SegNet
   └── Attention U-Net
   ↓
Lesion Segmentation Mask
   ↓
Quantitative Evaluation
   ↓
Visualization
   ↓
Web Demonstration
```

Mục tiêu cuối cùng không chỉ là tạo ra một mô hình có kết quả tốt, mà là xây dựng được một **quy trình nghiên cứu có thể kiểm chứng, tái lập và trình bày được bằng bằng chứng thực nghiệm**.

---

# 4. Scientific Goal

Mục tiêu khoa học trung tâm của đề tài là:

> **Đánh giá khả năng của cơ chế Attention trong kiến trúc U-Net đối với bài toán phân đoạn tổn thương phổi trên ảnh CT, thông qua việc so sánh Attention U-Net với U-Net tiêu chuẩn và các mô hình đối chứng trên cùng điều kiện thực nghiệm.**

Nghiên cứu phải tập trung vào **bằng chứng thực nghiệm**, thay vì chỉ dựa trên giả định rằng Attention U-Net sẽ tốt hơn U-Net.

---

# 5. Research Objectives

## OBJ-01 — Xây dựng pipeline dữ liệu CT

Xây dựng quy trình:

```text
Raw CT Data
→ Data Filtering
→ CT Preprocessing
→ HU Windowing
→ Normalization
→ Resize
→ Mask Processing
→ Dataset Split
→ DataLoader
```

Pipeline phải đảm bảo tính nhất quán giữa:

- Training set.
- Validation set.
- Test set.

---

## OBJ-02 — Xây dựng mô hình U-Net baseline

Triển khai U-Net tiêu chuẩn làm baseline.

U-Net được sử dụng làm cơ sở để:

- Thiết lập baseline performance.
- Kiểm tra pipeline huấn luyện.
- So sánh với Attention U-Net.
- Phân tích tác động của Attention Gates.

---

## OBJ-03 — Xây dựng mô hình SegNet

Triển khai SegNet như một mô hình đối chứng nhằm mở rộng việc đánh giá khả năng phân đoạn.

SegNet được sử dụng để so sánh:

- Khả năng phân đoạn.
- Khả năng bảo toàn biên tổn thương.
- Hiệu quả trên các vùng tổn thương có kích thước khác nhau.

---

## OBJ-04 — Xây dựng Attention U-Net

Triển khai Attention U-Net với **Attention Gates** trên các skip connections của U-Net.

Mục tiêu là kiểm tra liệu cơ chế Attention có giúp mô hình:

- Tập trung tốt hơn vào vùng tổn thương.
- Giảm ảnh hưởng của vùng nền không liên quan.
- Cải thiện chất lượng segmentation.
- Cải thiện khả năng nhận diện tổn thương có hình dạng hoặc kích thước phức tạp.

---

## OBJ-05 — Đánh giá và xây dựng prototype

Đánh giá các mô hình bằng các metric:

- Dice Similarity Coefficient (DSC / Dice).
- Intersection over Union (IoU / Jaccard).
- Precision.
- Recall.

Đồng thời xây dựng prototype web cho phép:

```text
Upload CT Image
       ↓
Model Inference
       ↓
Segmentation Mask
       ↓
Visualization
```

Prototype có mục đích **minh họa khả năng của mô hình nghiên cứu**, không phải hệ thống chẩn đoán y tế chính thức.

---

# 6. Research Questions

## RQ1

> **Attention Gates có cải thiện đáng kể kết quả Dice của mô hình U-Net đối với vùng tổn thương dạng Ground-Glass Opacity (GGO) so với U-Net tiêu chuẩn hay không?**

### Cần kiểm chứng bằng:

- U-Net baseline.
- Attention U-Net.
- Cùng dataset split.
- Cùng preprocessing.
- Điều kiện training tương đương.
- Cùng test set.
- Dice và các metric bổ sung.

---

## RQ2

> **SegNet hay U-Net có khả năng bảo toàn biên của các tổn thương nhỏ tốt hơn trên ảnh CT phổi?**

### Cần kiểm chứng bằng:

- Quantitative metrics.
- Qualitative visualization.
- Phân tích các trường hợp tổn thương nhỏ.
- So sánh prediction với Ground Truth.

Không được kết luận RQ2 chỉ dựa trên một vài hình ảnh minh họa.

---

# 7. Scope

## 7.1. In Scope

Đề tài bao gồm:

- Ảnh CT phổi từ dataset công khai.
- Dữ liệu 2D CT slices được trích xuất từ dữ liệu 3D.
- Preprocessing ảnh CT.
- HU windowing.
- Normalization.
- Resize ảnh.
- Binary segmentation mask.
- Data augmentation.
- U-Net.
- SegNet.
- Attention U-Net.
- BCE + Dice Loss.
- Model training.
- Model validation.
- Model testing.
- Dice.
- IoU.
- Precision.
- Recall.
- Qualitative analysis.
- Failure-case analysis.
- Web demonstration.
- Báo cáo khoa học.
- Presentation và defense.

---

## 7.2. Out of Scope

Các nội dung sau không thuộc phạm vi nghiên cứu chính:

- Chẩn đoán bệnh nhân thực tế.
- Thay thế bác sĩ trong quyết định y khoa.
- Triển khai hệ thống production tại bệnh viện.
- Xây dựng hệ thống PACS hoàn chỉnh.
- Phân tích toàn bộ bệnh lý phổi.
- Xây dựng mô hình foundation model.
- Nghiên cứu 3D segmentation chuyên sâu.
- Clinical validation trên bệnh nhân thật.
- Đánh giá hiệu quả điều trị.

Nếu phát sinh yêu cầu mới thuộc Out of Scope, phải được xem xét theo cơ chế **Scope Change Control**.

---

# 8. Dataset Scope

## 8.1. Dataset

Nguồn dữ liệu dự kiến bao gồm các dataset CT công khai được nêu trong đề cương nghiên cứu, trong đó có:

- COVID-19 CT Lung and Infection Segmentation Dataset.
- Một subset phù hợp từ LIDC-IDRI / nguồn MedData tương ứng.

Việc lựa chọn dataset thực tế phải dựa trên:

- Khả năng truy cập.
- Chất lượng annotation.
- Tính phù hợp với bài toán.
- Khả năng xử lý trong thời gian 8 tuần.
- Tính nhất quán của dữ liệu.

---

## 8.2. Dataset Size

Mục tiêu dự kiến:

> **1.500–2.000 2D CT slices**

Đây là mục tiêu nghiên cứu dự kiến, không được hiểu là số lượng bắt buộc nếu chất lượng dữ liệu hoặc annotation không đáp ứng.

---

## 8.3. Dataset Split

Dự kiến:

```text
70% — Training
15% — Validation
15% — Test
```

Quan trọng:

> Dataset phải được split theo **patient / volume level**, không được split ngẫu nhiên từng slice nếu điều đó gây ra data leakage giữa các tập.

---

# 9. Model Scope

## 9.1. Baseline

### U-Net

Vai trò:

> Baseline chính của nghiên cứu.

Mục đích:

- Thiết lập performance baseline.
- Làm cơ sở so sánh với Attention U-Net.
- Phân tích tác động của Attention mechanism.

---

## 9.2. Comparative Model

### SegNet

Vai trò:

> Mô hình đối chứng bổ sung.

Mục đích:

- So sánh với kiến trúc segmentation khác.
- Phân tích khả năng bảo toàn vùng/biên tổn thương.

---

## 9.3. Research Model

### Attention U-Net

Vai trò:

> Mô hình nghiên cứu trung tâm.

Attention Gates được tích hợp vào skip connections nhằm giúp mô hình chọn lọc các feature liên quan trước khi truyền thông tin sang decoder.

---

# 10. Experimental Principles

Đây là các nguyên tắc bắt buộc trong toàn bộ quá trình nghiên cứu.

## EXP-PRINCIPLE-01 — Không sửa Test Set

Test set không được sử dụng để:

- Tune hyperparameters.
- Chọn checkpoint.
- Điều chỉnh preprocessing dựa trên kết quả.
- Chọn mô hình sau khi xem kết quả.

---

## EXP-PRINCIPLE-02 — Không thay đổi Dataset Split tùy tiện

Một khi dataset split chính thức đã được xác định:

> Không được thay đổi split chỉ để cải thiện kết quả.

Nếu bắt buộc thay đổi, phải:

1. Ghi rõ lý do.
2. Tạo version split mới.
3. Ghi lại Experiment ID.
4. Chạy lại các thí nghiệm liên quan.

---

## EXP-PRINCIPLE-03 — Fair Comparison

Khi so sánh các mô hình:

- Dataset phải giống nhau.
- Test set phải giống nhau.
- Preprocessing phải tương đương.
- Metric phải giống nhau.
- Điều kiện thực nghiệm phải được ghi nhận.

---

## EXP-PRINCIPLE-04 — Không Cherry-picking

Không được chỉ lựa chọn các kết quả đẹp để đưa vào báo cáo.

Các trường hợp:

- Model thất bại.
- False Positive.
- False Negative.
- Segmentation sai biên.
- Tổn thương nhỏ bị bỏ sót.

phải được xem xét trong phần phân tích failure cases khi phù hợp.

---

## EXP-PRINCIPLE-05 — Experiment ID

Mọi thí nghiệm chính thức phải có:

```text
Experiment ID
```

Ví dụ:

```text
EXP-001
EXP-002
EXP-003
...
```

Nguyên tắc:

> **Không có Experiment ID → Không đưa kết quả vào Paper.**

---

# 11. Evaluation Framework

## 11.1. Quantitative Metrics

### Dice Similarity Coefficient

Đánh giá mức độ overlap giữa prediction và ground truth.

### IoU / Jaccard

Đánh giá mức độ giao nhau giữa hai vùng segmentation.

### Precision

Đánh giá tỷ lệ vùng được dự đoán là lesion thực sự thuộc lesion.

### Recall

Đánh giá khả năng phát hiện các vùng lesion trong Ground Truth.

---

## 11.2. Qualitative Evaluation

Các kết quả nên được trực quan hóa theo cấu trúc:

```text
Original CT
     ↓
Ground Truth
     ↓
Prediction
     ↓
Overlay
```

Mục tiêu:

- Quan sát vùng tổn thương.
- Quan sát biên.
- Phát hiện false positive.
- Phát hiện false negative.
- Phân tích các trường hợp tổn thương nhỏ.

---

# 12. Success Criteria

Dự án được xem là đạt mục tiêu khi hoàn thành được các điều kiện chính:

### Research

- [ ] Dataset được xử lý và documented.
- [ ] Dataset split được xác định.
- [ ] U-Net được triển khai.
- [ ] SegNet được triển khai.
- [ ] Attention U-Net được triển khai.
- [ ] Các mô hình được train/evaluate.
- [ ] Dice được tính toán.
- [ ] IoU được tính toán.
- [ ] Precision được tính toán.
- [ ] Recall được tính toán.
- [ ] Có qualitative visualization.
- [ ] Có failure-case analysis.
- [ ] RQ1 được trả lời bằng evidence.
- [ ] RQ2 được trả lời bằng evidence.

### Engineering

- [ ] Code có cấu trúc rõ ràng.
- [ ] Training pipeline có thể chạy lại.
- [ ] Configuration được lưu.
- [ ] Experiment ID được quản lý.
- [ ] Checkpoint được quản lý.
- [ ] Dataset version được ghi nhận.

### Demonstration

- [ ] Có prototype web.
- [ ] Có input CT.
- [ ] Có inference.
- [ ] Có segmentation visualization.

### Scientific Report

- [ ] Methodology hoàn chỉnh.
- [ ] Experimental setup rõ ràng.
- [ ] Results có bảng so sánh.
- [ ] Figures có caption.
- [ ] References được quản lý.
- [ ] Conclusions trả lời Research Questions.

---

# 13. Deliverables

| ID  | Deliverable         | Nội dung                      |
| --- | ------------------- | ----------------------------- |
| D01 | Project Charter     | Định hướng và phạm vi dự án   |
| D02 | Research Roadmap    | Kế hoạch 8 tuần               |
| D03 | Research Guidelines | Quy chuẩn nghiên cứu          |
| D04 | Dataset Pipeline    | Pipeline xử lý dữ liệu        |
| D05 | U-Net               | Baseline implementation       |
| D06 | SegNet              | Comparative implementation    |
| D07 | Attention U-Net     | Research model                |
| D08 | Experiment Logs     | Lịch sử thí nghiệm            |
| D09 | Evaluation Results  | Metrics và bảng kết quả       |
| D10 | Failure Cases       | Phân tích trường hợp thất bại |
| D11 | Web Demo            | Prototype minh họa            |
| D12 | Scientific Paper    | Bài báo/báo cáo nghiên cứu    |
| D13 | Presentation        | Slide trình bày               |
| D14 | Defense Materials   | Tài liệu chuẩn bị bảo vệ      |

---

# 14. Team Structure

Dự án gồm 5 thành viên. Các thành viên được xem là **ngang hàng về trách nhiệm chuyên môn và thực hiện công việc**.

Việc phân vai nhằm xác định **ownership**, tránh bỏ sót công việc và giúp phối hợp hiệu quả; không thể hiện cấp bậc quản lý trong nhóm.

---

## M1 — Research Coordination & Paper

### Primary Responsibilities

- Điều phối tiến độ nghiên cứu.
- Theo dõi WBS và roadmap.
- Điều phối các thành viên.
- Tổng hợp kết quả nghiên cứu.
- Quản lý cấu trúc tài liệu nghiên cứu.
- Tổng hợp và biên tập paper.
- Điều phối preparation cho presentation/defense.

### Scientific Responsibilities

- Theo dõi Research Questions.
- Đảm bảo kết quả cuối cùng trả lời đúng RQ.
- Kiểm tra tính nhất quán giữa Methodology → Experiment → Results → Conclusion.

> M1 là đầu mối điều phối dự án, nhưng không được hiểu là cấp trên của các thành viên khác.

---

## M2 — Data & Evaluation

### Responsibilities

- Dataset.
- Data preprocessing.
- Dataset split.
- Ground Truth validation.
- Evaluation pipeline.
- Metrics.
- Visualization.
- Failure-case analysis.

---

## M3 — Core AI Engineering

### Responsibilities

- U-Net.
- Attention U-Net.
- Training pipeline.
- Loss function.
- Hyperparameters.
- Model checkpoint.
- Training experiments.
- Model debugging.

---

## M4 — Baseline & Demo Engineering

### Responsibilities

- SegNet.
- Baseline comparison.
- Inference pipeline.
- Web prototype.
- Visualization interface.
- Integration model → demo.

---

## M5 — Research QA & Reproducibility

### Responsibilities

- Experiment tracking.
- Reproducibility checking.
- Configuration verification.
- Dataset split verification.
- Code review hỗ trợ.
- Evidence validation.
- Kiểm tra tính nhất quán giữa kết quả thực nghiệm và báo cáo.
- Kiểm tra repository trước khi đóng milestone.
- Hỗ trợ kiểm tra paper trước khi submission.

---

# 15. Governance

## 15.1. Work Assignment

Mỗi task phải có:

```text
Task ID
Owner
Contributors
Deadline
Dependency
Deliverable
Acceptance Criteria
Evidence
Status
```

Ví dụ:

```text
Task ID: MODEL-001
Owner: M3
Contributors: M5
Deadline: Week 3
Deliverable: U-Net baseline
Evidence: checkpoint + training log + evaluation result
Status: REVIEW
```

---

# 16. Definition of Done

Một task chỉ được đánh dấu:

```text
DONE
```

khi đáp ứng:

```text
Implementation
      +
Validation
      +
Evidence
      +
Documentation
      =
DONE
```

Không được đánh dấu DONE chỉ vì:

- Code đã viết.
- Code chạy được.
- Thành viên đã hoàn thành phần việc.
- Có kết quả nhưng chưa lưu evidence.

---

# 17. Weekly Coordination

Mỗi tuần cần xác định:

```text
WEEK GOAL
     ↓
TASKS
     ↓
OWNERS
     ↓
DEPENDENCIES
     ↓
EVIDENCE
     ↓
WEEKLY REVIEW
     ↓
NEXT WEEK
```

Weekly review tập trung vào:

1. Đã hoàn thành gì?
2. Evidence nằm ở đâu?
3. Có task nào bị block?
4. Có dependency nào ảnh hưởng thành viên khác?
5. Có kết quả nào cần kiểm tra lại?
6. Tuần tiếp theo cần ưu tiên gì?

---

# 18. Scope Change Control

Mọi yêu cầu mới phải được phân loại:

### Type A — Không ảnh hưởng Scope

Ví dụ:

- Refactor code.
- Sửa documentation.
- Cải thiện visualization.
- Sửa bug.

→ Có thể thực hiện trong phạm vi task.

### Type B — Ảnh hưởng Timeline nhưng không thay đổi Research Goal

Ví dụ:

- Thêm experiment.
- Thêm visualization.
- Điều chỉnh pipeline.

→ Phải ghi nhận ảnh hưởng đến timeline.

### Type C — Thay đổi Research Scope

Ví dụ:

- Thêm architecture mới.
- Đổi dataset chính.
- Chuyển từ 2D sang 3D segmentation.
- Thay đổi Research Question.

→ Phải được cả nhóm thống nhất trước khi thực hiện.

---

# 19. Risk Boundaries

## RISK-01 — Data Leakage

**Nguy cơ:** Slice của cùng một patient xuất hiện ở nhiều dataset split.

**Biện pháp:**

- Split theo patient/volume.
- Kiểm tra dataset ID.
- Document split.

---

## RISK-02 — Không đủ thời gian training

**Biện pháp:**

- Sử dụng GPU phù hợp.
- Chuẩn hóa input size.
- Theo dõi training time.
- Ưu tiên experiment phục vụ trực tiếp RQ.

---

## RISK-03 — Model không hội tụ

**Biện pháp:**

- Kiểm tra preprocessing.
- Kiểm tra mask.
- Kiểm tra loss.
- Kiểm tra learning rate.
- Kiểm tra sample visualization.

---

## RISK-04 — Kết quả không cải thiện với Attention

Nếu Attention U-Net không vượt U-Net:

> Không được thay đổi kết quả hoặc lựa chọn kết quả có lợi để tạo ra kết luận mong muốn.

Thay vào đó phải:

- Phân tích nguyên nhân.
- Kiểm tra experiment.
- Phân tích failure cases.
- Báo cáo kết quả thực tế.

Một kết quả không cải thiện vẫn là kết quả nghiên cứu có giá trị nếu thí nghiệm được thiết kế và thực hiện đúng.

---

# 20. 8-Week Project Timeline

## Week 1 — Research & Environment

### Main Goals

- Literature review.
- Dataset investigation.
- Environment setup.
- Zotero setup.
- Repository setup.
- Research documentation.

### Expected Outputs

- Research references.
- Environment ready.
- Dataset candidate identified.
- Initial project structure.

---

## Week 2 — Dataset & DataLoader

### Main Goals

- Download dataset.
- Filter data.
- Process CT.
- Process masks.
- Resize.
- Dataset split.
- Build DataLoader.

### Expected Outputs

- Dataset version.
- Dataset split.
- Preprocessing pipeline.
- DataLoader.

---

## Week 3 — Baseline Models

### Main Goals

- Implement U-Net.
- Implement SegNet.
- Configure BCE + Dice Loss.
- Train baseline.

### Expected Outputs

- U-Net checkpoint.
- SegNet checkpoint.
- Training logs.
- Initial evaluation.

---

## Week 4 — Attention U-Net

### Main Goals

- Implement Attention Gates.
- Build Attention U-Net.
- Train model.
- Record experiments.

### Expected Outputs

- Attention U-Net checkpoint.
- Experiment logs.
- Initial comparison.

---

## Week 5 — Evaluation & Failure Analysis

### Main Goals

- Evaluate all models.
- Calculate metrics.
- Generate visualizations.
- Analyze failure cases.
- Answer RQ1.
- Answer RQ2.

### Expected Outputs

- Metrics table.
- Visualization set.
- Failure cases.
- Preliminary research findings.

---

## Week 6 — Web Demo

### Main Goals

- Build Streamlit/Gradio prototype.
- Connect trained model.
- Implement inference.
- Display segmentation.

### Expected Outputs

```text
CT Input
→ Model
→ Mask
→ Overlay
```

---

## Week 7 — Paper

### Main Goals

- Methodology.
- Experimental setup.
- Results.
- Discussion.
- Conclusion.
- References.

### Expected Outputs

- Complete paper draft.
- Figures.
- Tables.
- References.

---

## Week 8 — Finalization & Defense

### Main Goals

- Final report.
- IEEE formatting.
- Slide preparation.
- Defense preparation.
- Repository cleanup.
- Final rehearsal.

### Expected Outputs

- Final report.
- Presentation.
- Demo.
- Git repository.
- Defense Q&A preparation.

---

# 21. Critical Path

Critical Path của dự án:

```text
Dataset
   ↓
Preprocessing
   ↓
Dataset Split
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
Presentation / Defense
```

Các thành viên cần đặc biệt chú ý những dependency này vì việc chậm ở một bước có thể ảnh hưởng trực tiếp đến các bước phía sau.

---

# 22. Research Completion

Dự án chỉ được xem là hoàn thành khi:

```text
Research Questions
        ↓
Experimental Design
        ↓
Implementation
        ↓
Experiments
        ↓
Evidence
        ↓
Analysis
        ↓
Conclusion
```

được liên kết một cách nhất quán.

Không coi dự án hoàn thành chỉ vì:

- Model chạy được.
- Có web demo.
- Có hình ảnh đẹp.
- Có một con số Dice cao.

Tiêu chí cuối cùng là:

> **Kết luận khoa học phải được hỗ trợ bởi evidence thực nghiệm có thể truy xuất.**

---

# 23. Core Team Rules

## Rule 01

> Không thay đổi Research Question tùy tiện.

## Rule 02

> Không thay đổi Test Set để cải thiện kết quả.

## Rule 03

> Không cherry-pick kết quả.

## Rule 04

> Không đưa kết quả không có Experiment ID vào Paper.

## Rule 05

> Không đánh dấu DONE nếu chưa có Evidence.

## Rule 06

> Không sử dụng dữ liệu Test để tune model.

## Rule 07

> Mọi thay đổi quan trọng phải được document.

## Rule 08

> Code, experiment và report phải nhất quán.

## Rule 09

> Mọi thành viên đều chịu trách nhiệm với phần việc của mình.

## Rule 10

> Kết quả nghiên cứu phải phản ánh kết quả thực tế, không điều chỉnh kết quả để phù hợp với giả thuyết ban đầu.

---

# 24. Definition of Done — Project Level

Project được xem là **DONE** khi tất cả điều kiện sau được đáp ứng:

```text
[✓] Dataset pipeline hoàn chỉnh
[✓] Dataset split hợp lệ
[✓] U-Net hoàn chỉnh
[✓] SegNet hoàn chỉnh
[✓] Attention U-Net hoàn chỉnh
[✓] Experiments có Experiment ID
[✓] Evaluation hoàn chỉnh
[✓] Failure analysis hoàn chỉnh
[✓] RQ1 được trả lời
[✓] RQ2 được trả lời
[✓] Web demo hoạt động
[✓] Scientific report hoàn chỉnh
[✓] Figures & Tables hoàn chỉnh
[✓] References hoàn chỉnh
[✓] Presentation hoàn chỉnh
[✓] Defense preparation hoàn chỉnh
[✓] Repository được kiểm tra
```

---

# 25. Source of Truth

Khi có thông tin không thống nhất giữa các tài liệu, ưu tiên theo thứ tự:

```text
01_PROJECT_CHARTER.md
        ↓
02_WBS_AND_ROADMAP.md
        ↓
03_RESEARCH_GUIDELINES.md
        ↓
04_TECH_STACK_AND_DATASET.md
        ↓
05_OPERATIONAL_MANUAL.md
        ↓
06_PRESENTATION_DEFENSE_KIT.md
        ↓
07_EXPERIMENT_AND_EVIDENCE_LOG.md
        ↓
Source Code / Experiment Artifacts
```

Tuy nhiên, **Experiment Artifact là nguồn bằng chứng thực tế cho kết quả thực nghiệm**.

Ví dụ:

```text
Paper nói Dice = 0.82
        ↓
Experiment Log
        ↓
Checkpoint
        ↓
Evaluation Output
```

Nếu không truy xuất được evidence, kết quả phải được kiểm tra lại trước khi sử dụng trong báo cáo.

---

# 26. Document Dependencies

```text
01_PROJECT_CHARTER
        │
        ├── 02_WBS_AND_ROADMAP
        │
        ├── 03_RESEARCH_GUIDELINES
        │
        ├── 04_TECH_STACK_AND_DATASET
        │
        ├── 05_OPERATIONAL_MANUAL
        │
        ├── 06_PRESENTATION_DEFENSE_KIT
        │
        └── 07_EXPERIMENT_AND_EVIDENCE_LOG
```

Trong đó:

- `01_PROJECT_CHARTER.md` xác định **Why / What**.
- `02_WBS_AND_ROADMAP.md` xác định **When / Who**.
- `03_RESEARCH_GUIDELINES.md` xác định **How scientifically**.
- `04_TECH_STACK_AND_DATASET.md` xác định **With what**.
- `05_OPERATIONAL_MANUAL.md` xác định **How operationally**.
- `06_PRESENTATION_DEFENSE_KIT.md` xác định **How to present and defend**.
- `07_EXPERIMENT_AND_EVIDENCE_LOG.md` xác định **What evidence has been produced**.

---

# 27. Document Control

| Field            | Value                              |
| ---------------- | ---------------------------------- |
| Document         | `01_PROJECT_CHARTER.md`            |
| Version          | 1.1                                |
| Status           | Baseline                           |
| Project Duration | 8 weeks                            |
| Team Size        | 5                                  |
| Coordination     | M1 – Research Coordination & Paper |
| Review Frequency | Weekly                             |
| Major Revision   | Requires team agreement            |

### Version History

| Version | Change                                                       | Reason                                          |
| ------- | ------------------------------------------------------------ | ----------------------------------------------- |
| 1.0     | Initial Project Charter                                      | Establish project baseline                      |
| 1.1     | Thay đổi cách thể hiện vai trò M1, bỏ chức danh Project Lead | Đảm bảo 5 thành viên ngang hàng trong công việc |

---

# 28. Final Project Principle

Toàn bộ dự án được vận hành theo nguyên tắc:

> **Clear Scope → Controlled Experiment → Traceable Evidence → Reproducible Result → Scientific Conclusion**

Hay nói ngắn gọn:

> **Không nghiên cứu theo cảm tính. Không quản lý theo trí nhớ. Không kết luận khi chưa có bằng chứng.**

Mỗi thành viên chịu trách nhiệm với phần việc của mình; nhóm chịu trách nhiệm với kết quả chung; M1 là đầu mối điều phối để đảm bảo các phần việc được kết nối thành một quy trình nghiên cứu thống nhất.
