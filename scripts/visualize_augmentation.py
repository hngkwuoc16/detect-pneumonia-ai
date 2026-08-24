import sys
sys.path.append('d:/detect-pneumonia-ai')  # nếu chạy từ thư mục scripts

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transform

def denormalize(tensor):
    """Chuyển tensor đã normalize về dạng hiển thị được (H,W,C uint8)."""
    # tensor shape (3,H,W)
    img = tensor.cpu().numpy().transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean   #do normalize có công thức: (img - mean) / std, nên để denormalize thì img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def main():
    # Đọc một ảnh từ train (thay đường dẫn thật)
    csv_path = 'data/splits.csv'  # sửa lại nếu cần
    df = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'train'].head(1)  # lấy 1 ảnh

    transform = get_train_transform(image_size=224)

    # Áp dụng transform nhiều lần trên cùng ảnh
    for i in range(6):
        dataset = ChestXrayDataset(train_df, transform)
        img_tensor, _ = dataset[0]
        img_vis = denormalize(img_tensor)

        plt.subplot(2,3,i+1)
        plt.imshow(img_vis)
        plt.title(f'Aug {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()