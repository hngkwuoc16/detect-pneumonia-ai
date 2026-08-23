[ROADMAP TỔNG QUAN (12 PHASES)]
# Explainable AI Pneumonia Classification
End-to-End Research & Prototype Development Roadmap
- Architecture: PyTorch & PyYAML
- Target Task: Medical Image Binary Class
- Explainability: Grad-CAM & Error Analysis

---

### PHASE 0: Project Setup
- Description: Môi trường Virtualenv, cấu trúc thư mục modular (src/), Git workflow, cài đặt dependencies, và hàm cố định set_seed() đảm bảo tính tái lập.
- Key Deliverables: setup.py, requirements.txt, configs/config.yaml, src/utils.py

↓

### PHASE 1: Dataset Validation
- Description: Kiểm tra tính hợp lệ của ảnh X-quang, phát hiện file bị lỗi hỏng, thống kê phân phối nhãn (Normal vs Pneumonia) và chia tập dữ liệu Train/Val/Test độc lập.
- Key Deliverables: notebooks/01_dataset_validation.ipynb, data/splits.csv

↓

### PHASE 2: Preprocessing
- Description: Chuẩn hóa kích thước ảnh về 224x224, chuyển đổi Tensor và áp dụng Normalization dựa trên mean/std chuẩn của ImageNet hoặc dataset gốc.
- Key Deliverables: Custom PyTorch Dataset class, DataLoader pipeline

↓

### PHASE 3: Augmentation
- Description: Tăng cường dữ liệu bằng Albumentations (xoay nhẹ, chỉnh độ tương phản, ShiftScaleRotate) nhằm tránh overfitting mà vẫn giữ nguyên đặc trưng bệnh lý.
- Key Deliverables: src/dataset.py với tích hợp Albumentations transforms

↓

### PHASE 4: Baseline Model
- Description: Xây dựng kiến trúc mô hình phân loại dựa trên ResNet/DenseNet (Transfer Learning), nạp Pretrained Weights và thay thế lớp Classifier cuối.
- Key Deliverables: src/models.py, kiểm tra Tensor forward pass với dummy input

↓

### PHASE 5: Training
- Description: Thiết lập Training Loop với BCEWithLogitsLoss/CrossEntropy, Adam/AdamW Optimizer, Learning Rate Scheduler, Early Stopping và lưu Checkpoint tốt nhất.
- Key Deliverables: src/train.py, lưu vết log loss & accuracy qua các epochs

↓

### PHASE 6: Validation & Threshold Selection
- Description: Đánh giá mô hình trên tập Validation, vẽ đường cong Precision-Recall / ROC, tối ưu hóa ngưỡng xác suất (Threshold Tuning) để tăng Recall cho lớp Pneumonia.
- Key Deliverables: notebooks/02_threshold_tuning.ipynb, optimal threshold value

↓

### PHASE 7: Test & Comprehensive Evaluation
- Description: Đánh giá độc lập mô hình trên tập Test (chưa từng thấy): Tính toán Confusion Matrix, Accuracy, Precision, Recall, F1-Score, và AUC-ROC Score.
- Key Deliverables: src/evaluate.py, Báo cáo chỉ số đánh giá tổng thể

↓

### PHASE 8: Error Analysis
- Description: Trích xuất các trường hợp dự đoán sai (False Positives & False Negatives) để phân tích nguyên nhân (ảnh mờ, nhiễu thiết bị, dấu hiệu tổn thương không rõ).
- Key Deliverables: notebooks/03_error_analysis.ipynb, Bảng tổng hợp mẫu lỗi

↓

### PHASE 9: Grad-CAM / Explainability
- Description: Áp dụng kỹ thuật Grad-CAM / Grad-CAM++ để tạo Heatmap giải thích vùng ảnh X-quang trọng yếu mà mô hình tập trung vào khi đưa ra chẩn đoán.
- Key Deliverables: src/explainability.py, Heatmap visualization overlays

↓

### PHASE 10: Controlled Experiments
- Description: Thực hiện các thực nghiệm so sánh: ResNet vs DenseNet, ảnh hưởng của Data Augmentation, thử nghiệm Weighted Loss giải quyết imbalanced data.
- Key Deliverables: Bảng so sánh thực nghiệm (Ablation Study)

↓

### PHASE 11: Documentation & Research Analysis
- Description: Hoàn thiện tài liệu dự án, tổng hợp báo cáo nghiên cứu, đóng gói mô hình và viết hướng dẫn tái lập kết quả (Reproducibility Guide).
- Key Deliverables: README.md hoàn chỉnh, Báo cáo nghiên cứu PDF
