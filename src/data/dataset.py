import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging


logger = logging.getLogger(__name__)

#dataset: nhiệm vụ để đọc ảnh từ đường dẫn (bằng cv2), áp dụng các tính năng transform, trả về tensor ảnh và nhãn cho model học
class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset cho ảnh X-quang phổi.
    Đọc ảnh grayscale, chuyển sang 3 kênh, áp dụng transform.
    """

    def __init__(self, df, transform=None, error_log_path=None, label_smoothing=0.0):
        """
        Args:
            df: DataFrame chứa ít nhất cột 'path' (đường dẫn ảnh) và 'label' (NORMAL/PNEUMONIA)
            transform: Albumentations Compose
            error_log_path: Tệp tùy chọn để lưu các ảnh không đọc được.
            label_smoothing: Tỷ lệ label smoothing.
        """
        if transform is None:
            raise ValueError(
                "transform bắt buộc phải được cung cấp. "
                "Hãy dùng get_transforms(split, image_size) từ src.data.transforms."
            )
        if not callable(transform):
            raise TypeError("transform phải là một callable Albumentations transform.")

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self.failed_samples = []
        self.label_smoothing = label_smoothing #bổ sung label smoothing để tránh overfitting do label cứng
        #có 4 thuộc tính của code: df: các ảnh, transform: các phép biến đổi từ module transforms.py
        #error_log_path: đường dẫn tệp lỗi, failed_samples: danh sách các mẫu lỗi

    def __len__(self):
        return len(self.df)

    def _smooth_label(self, label):
        """Áp dụng label smoothing cho nhãn binary (0 hoặc 1)."""
        # label: 0 hoặc 1
        if self.label_smoothing > 0:
            return label * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = 1 if row['label'] == 'PNEUMONIA' else 0  # 0: NORMAL, 1: PNEUMONIA
        label = self._smooth_label(label)

        # Đọc ảnh dưới dạng grayscale (1 kênh)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        except cv2.error as error:
            self._record_failed_sample(img_path, f"OpenCV error: {error}")
            return None  #raise lỗi đồng thời trả về None để tránh lỗi ngừng chương trình trong quá trình đọc batch

        if img is None:
            self._record_failed_sample(img_path, "Ảnh không tồn tại hoặc không đọc được")
            return None #tránh gây ngừng chương trình trong lúc đọc ảnh hoặc batch

        # Chuyển grayscale -> RGB 3 kênh (nhân bản kênh xám)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  #từ BGR -> RGB

        transformed = self.transform(image=img)
        img_tensor = transformed['image']

        # Trả về tensor ảnh (3, 224, 224) và nhãn (0 hoặc 1)
        return img_tensor, torch.tensor(label, dtype=torch.float32)  #Đầu vào hàm mất mát phải ứng với float32, nếu ko khúc này phải chỉnh lại để phù hợp kiểu dữ liệu

    def _record_failed_sample(self, img_path, reason):
        """Ghi nhận lỗi nhưng không làm hỏng toàn bộ DataLoader worker."""
        failure = {"path": str(img_path), "reason": reason}
        self.failed_samples.append(failure)
        logger.warning("Bỏ qua ảnh lỗi: %s (%s)", img_path, reason)

        if self.error_log_path is not None:
            self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.error_log_path.open("a", encoding="utf-8") as file:
                file.write(f"{img_path}\t{reason}\n")   #ghi nhận lại thông tin lỗi


def skip_failed_samples(batch):
    """Collate function loại mẫu lỗi; batch toàn mẫu lỗi trả về None."""
    valid_samples = [sample for sample in batch if sample is not None]
    if not valid_samples:
        return None

    images, labels = zip(*valid_samples)
    return torch.stack(images), torch.stack(labels)