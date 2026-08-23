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