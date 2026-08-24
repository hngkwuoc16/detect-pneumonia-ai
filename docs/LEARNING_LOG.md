# Learning Log

## Phase 1: Dataset Validation

### 1. Aspect ratio trong ảnh X-quang
- Là tỷ lệ width/height. Ảnh X-quang phổi chuẩn thường có ratio ~1.0–1.5. Ratio quá cao thường do crop mất thông tin.
- Cần kiểm tra thủ công để chọn ngưỡng loại bỏ hợp lý (cân bằng giữa loại bỏ ảnh xấu và giữ ảnh tốt).

### 2. Blur detection
- Không nên đo trực tiếp trên ảnh gốc vì nền đen gây nhiễu.
- Dùng Otsu threshold để tách foreground, crop bỏ nền.
- CLAHE cải thiện tương phản cục bộ, giúp so sánh công bằng hơn.
- Laplacian variance: biến thiên của đạo hàm bậc hai, ảnh nét có giá trị cao.
- Tenengrad: độ lớn gradient trung bình, cũng đo sắc nét.
- Kết hợp log scale để giảm ảnh hưởng của outlier.

### 3. Perceptual hash (pHash)
- Hash ảnh dựa trên DCT, phản ánh cấu trúc ảnh thay vì pixel chính xác.
- hash_size=16 → 256 bit, tăng độ phân biệt.
- Hamming distance đo số bit khác nhau giữa hai hash.
- Khảo sát histogram Hamming distance để chọn ngưỡng trùng lặp.

### 4. Stratified split
- Dùng train_test_split với stratify để giữ tỷ lệ lớp khi chia dữ liệu.
- Quan trọng khi dataset mất cân bằng (Normal vs Pneumonia ~1:2.4).

## Phase 2: Preprocessing

### 5. Letterbox resize (aspect ratio + padding)
- Giữ nguyên tỷ lệ ảnh, pad thêm viền đen để đạt kích thước vuông.
- Dùng albumentations.LongestMaxSize và PadIfNeeded.

### 6. Normalization
- ImageNet stats (mean/std) dùng cho pretrained models.
- Vì ảnh grayscale nhưng nhân bản 3 kênh, stats vẫn áp dụng được.

### 7. PyTorch Dataset & DataLoader
- Custom Dataset kế thừa torch.utils.data.Dataset, cần __len__, __getitem__.
- DataLoader quản lý batch, shuffle, num_workers.

## Phase 3: Augmentation

### 8. Các phép augmentation cho ảnh y tế
- Cần chọn phép biến đổi không phá vỡ cấu trúc giải phẫu: tránh VerticalFlip, biến dạng đàn hồi, crop bỏ vùng ngoại vi.
- Augmentation cường độ (độ sáng, tương phản, gamma) mô phỏng sự khác biệt máy chụp, giúp mô hình bền vững.
- RandomErasing (CoarseDropout) với khối nhỏ giúp mô hình không học vẹt, phù hợp bài toán classification.

### 9. Label smoothing
- Kỹ thuật regular hóa nhãn: thay nhãn cứng 0/1 bằng nhãn mềm (0.05/0.95). Giúp mô hình không quá tự tin, cải thiện calibration.
- Triển khai trong Dataset: `label = label * (1 - smoothing) + 0.5 * smoothing`.

### 10. Albumentations API mới
- Khi dùng phiên bản ≥2.0, cần cập nhật tham số: `fill`, `std_range`, `num_holes_range`.
- Lưu ý kiểm tra warnings để điều chỉnh.