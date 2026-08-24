# Decision Log

## Phase 1: Dataset Validation

### 1. Loại bỏ ảnh có aspect ratio >= 1.85
- **Lý do**: Qua khảo sát thủ công, các ảnh có tỷ lệ khung hình >1.85 thường bị crop mất thông tin giải phẫu (đỉnh phổi, góc sườn). Ảnh trẻ em có ratio ~2 vẫn tốt nhưng số lượng ít, chấp nhận hy sinh để đảm bảo chất lượng chung.
- **Phương pháp thay thế**: Không dùng ngưỡng cứng tự động, kết hợp kiểm tra thủ công.

### 2. Sử dụng blur detection với foreground crop + CLAHE + Laplacian variance + Tenengrad
- **Lý do**: Ảnh X-quang có nền đen lớn, nếu không crop foreground sẽ làm sai lệch blur score. CLAHE giúp chuẩn hóa tương phản, tránh ảnh tối bị coi là mờ. Kết hợp hai chỉ số (Laplacian variance và Tenengrad) giúp đánh giá sắc nét toàn diện hơn.
- **Ngưỡng**: Chọn quantile 5% blur_score (tương đương loại 5% ảnh mờ nhất).

### 3. Dùng pHash 256-bit với Hamming distance threshold = 10
- **Lý do**: pHash 256-bit tăng độ phân biệt; khảo sát histogram khoảng cách Hamming cho thấy phân bố lệch, không có valley rõ ràng, nên chọn ngưỡng theo mức sát mép trái. Qua khảo sát 10 là threshold ổn.
- **So sánh toàn bộ ảnh tốt (không phân biệt split)**: Phát hiện trùng lặp giữa mọi split, sau đó mới chia lại train/val/test.

### 4. Chia lại split với tỷ lệ 70/15/15 stratified
- **Lý do**: Tập validation gốc của Kaggle quá nhỏ (16 ảnh), không đủ tin cậy. Gộp toàn bộ ảnh sạch rồi chia ngẫu nhiên có kiểm soát (stratified) giúp đảm bảo cân bằng nhãn và đủ lớn.

## Phase 2: Preprocessing

### 5. Resize giữ aspect ratio + padding (letterbox)
- **Lý do**: Ảnh y tế cần bảo toàn cấu trúc giải phẫu, tránh méo. Sau khi lọc aspect ratio <1.85, vùng padding không quá lớn. Dùng cv2.BORDER_CONSTANT (pad màu đen) và albumentations.LongestMaxSize + PadIfNeeded.

### 6. Dùng ImageNet stats để normalize
- **Lý do**: Tận dụng pretrained models (ResNet/DenseNet) quen với chuẩn hóa ImageNet. Về sau có thể thử custom stats trong các thí nghiệm.

### 7. Đọc grayscale rồi nhân bản 3 kênh
- **Lý do**: Đảm bảo mọi ảnh đều là grayscale thật sự, tránh nhiễu màu từ scanner. Pretrained models yêu cầu input 3 kênh.

## Phase 3: Augmentation

### 8. Chọn các phép augmentation an toàn
- Sử dụng HorizontalFlip, Affine (shift ±5%, scale ±10%, rotate ±10°), RandomBrightnessContrast ±0.2, CLAHE, RandomGamma, GaussNoise nhẹ, CoarseDropout (khối đen ≤16x16, p=0.3, khối đen tương đối nhỏ tránh phủ đen vùng quan trọng trong chẩn đoán pneumonia).
- Lý do: Tránh biến dạng quá mức làm mất đặc trưng bệnh lý. Không dùng VerticalFlip, ElasticTransform, RandomCrop.
- Không dùng MixUp trong transforms vì sẽ triển khai trong training loop (Phase 5) với tỷ lệ 10%, alpha=0.2 do nguy cơ tạo ảnh "quái dị" khi trộn ảnh người lớn/trẻ em.
- Label smoothing = 0.1 áp dụng trong Dataset cho tập train.

### 9. Cập nhật API Albumentations ≥2.0
- Các tham số cũ `value`, `fill_value`, `var_limit` không còn dùng. Thay bằng `fill`, `std_range`, `num_holes_range`, `hole_height_range`, `hole_width_range`.