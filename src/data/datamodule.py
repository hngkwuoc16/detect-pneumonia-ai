import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_transforms

#datamodule: nhiệm vụ để phân phối, quản lý dữ liệu, đọc CSV, tạo DataLoader cho train/val/test để đẩy vào (theo batch) cho model học
#sinh ra để kết hợp các module: dataset, transforms kèm với dataloader thành một module duy nhất để quản lý dữ liệu và đẩy vào model học
class ChestXrayDataModule:
    """
    Quản lý dữ liệu: đọc CSV, tạo DataLoader cho train/val/test.
    """

    def __init__(self, csv_path, batch_size=32, num_workers=4, image_size=224):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def setup(self):
        """Đọc CSV và tạo transforms."""
        df = pd.read_csv(self.csv_path)

        # Lọc theo split
        self.train_df = df[df['split'] == 'train'].reset_index(drop=True)
        self.val_df = df[df['split'] == 'val'].reset_index(drop=True)
        self.test_df = df[df['split'] == 'test'].reset_index(drop=True)

        # Tạo transforms
        self.train_transform = get_transforms('train', self.image_size)
        self.val_transform = get_transforms('val', self.image_size)
        self.test_transform = get_transforms('test', self.image_size)

    def train_dataloader(self):
        return DataLoader(
            ChestXrayDataset(self.train_df, self.train_transform),
            batch_size=self.batch_size,
            shuffle=True,  #xáo trộn để mô hình ko học vẹt
            num_workers=self.num_workers,  #mặc định lấy 4, có thể giảm nếu máy yếu
            pin_memory=False,  #tăng luồng dữ liệu từ CPU -> GPU, nếu máy ko có GPU thì bỏ qua
            #vì máy ko có GPU nên pin_memory=False để tránh lỗi, nếu có GPU thì pin_memory=True để tăng tốc độ truyền dữ liệu
        )

    def val_dataloader(self):
        return DataLoader(
            ChestXrayDataset(self.val_df, self.val_transform),
            batch_size=self.batch_size,
            shuffle=False,  #không cần shuffle vì là validation
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            ChestXrayDataset(self.test_df, self.test_transform),
            batch_size=self.batch_size,
            shuffle=False,   #test cũng ko cần shuffle
            num_workers=self.num_workers,
            pin_memory=False,
        )