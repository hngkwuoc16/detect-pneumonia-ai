# Project State

## Hiện tại: Đang ở Phase 4

### Đã hoàn thành
- Phase 0: Project Setup
  - Tạo cấu trúc thư mục, môi trường ảo, cài dependencies.
  - Viết src/utils.py (set_seed, logging...).
- Phase 1: Dataset Validation
  - Kiểm tra tính hợp lệ ảnh (không file lỗi).
  - Phân tích aspect ratio → loại bỏ ảnh có aspect ratio >= 1.85
  - Xây dựng blur detection (foreground crop + CLAHE + Laplacian variance + Tenengrad), chọn ngưỡng quantile 5%.
  - Loại bỏ ảnh trùng lặp dùng perceptual hash (pHash 256-bit), threshold Hamming distance = 10 (khảo sát các ngưỡng lệch trái và chọn 10 là phù hợp).
  - Chia lại tập train/val/test theo tỷ lệ 70/15/15 (stratified theo label).
  - Lưu kết quả vào data/splits.csv.
- Phase 2: Preprocessing
  - Đã tạo src/data/transforms.py (resize giữ aspect ratio + padding, normalize ImageNet stats).
  - Đã tạo src/data/dataset.py (đọc grayscale, nhân bản 3 kênh, áp transform).
  - Đã tạo src/data/datamodule.py (DataLoader cho train/val/test).
  - Đã kiểm tra pipeline bằng scripts/test_dataloader.py.
- Phase 3: Augmentation
  - Thêm các phép augmentation: HorizontalFlip, Affine (shift/scale/rotate nhẹ), RandomBrightnessContrast, CLAHE, RandomGamma, GaussNoise, CoarseDropout (RandomErasing).
  - Tách get_train_transform (có aug) và get_val_transform (không aug).
  - Áp dụng label smoothing = 0.1 cho tập train (val/test giữ nhãn cứng).
  - Cập nhật tham số mới cho Albumentations ≥2.0 (fill, std_range, num_holes_range...).
  - Đã kiểm tra trực quan augmentation, không có lỗi, ảnh vẫn giữ cấu trúc giải phẫu.

### Tiếp theo
- Chuyển sang Phase 4: Baseline Model

### Tình trạng dữ liệu 
#như phase 2
- File splits.csv: ~5300 ảnh sạch (sau khi lọc).
- Phân bố:
  - Train: ~70% (Normal ~?, Pneumonia ~?)
  - Val: ~15% (Normal ~?, Pneumonia ~?)
  - Test: ~15% (Normal ~?, Pneumonia ~?)