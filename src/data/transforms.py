import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224):
    """
    Transform cho tập train: có augmentation.
    """
    return A.Compose([
        # Resize cạnh dài về image_size, giữ nguyên aspect ratio
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_CUBIC),

        # Pad về đúng image_size x image_size (thêm viền đen)
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,          # <-- sửa từ value thành fill_value
        ),

        # --- Các phép augmentation ---
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),          # scale_limit=0.1
            translate_percent=(-0.05, 0.05),  # shift_limit=0.05
            rotate=(-10, 10),           # rotate_limit=10
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.CLAHE(
            clip_limit=1.5,
            tile_grid_size=(8, 8),
            p=0.5,
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3,
        ),
        A.GaussNoise(
            std_range = (0.02, 0.05),      # <-- thay var_limit bằng std_range (đơn vị pixel)
            p=0.3,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 1),      # <-- max_holes=1, min_holes=1
            hole_height_range=(8, 16),   # <-- min_height=8, max_height=16
            hole_width_range=(8, 16),    # <-- min_width=8, max_width=16
            fill=0,
            p=0.3,
        ),

        # Chuẩn hóa theo ImageNet stats
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # Chuyển sang tensor
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224):
    """
    Transform cho tập val/test: không augmentation.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_CUBIC),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,          # <-- sửa
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])