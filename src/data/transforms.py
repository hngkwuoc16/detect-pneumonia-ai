import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics (mean và std chuẩn hóa cho pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#transform: công cụ để preprocess ảnh đầu vào gồm: resize, normalize, thêm viền đen, ...
def get_transforms(split: str, image_size: int = 224):
    """
    Trả về transform phù hợp cho từng split.

    Args:
        split: 'train', 'val', hoặc 'test'
        image_size: kích thước ảnh đầu ra (cạnh dài nhất sau resize, pad về vuông)

    Returns:
        Albumentations Compose
    """
    # Cả train và val/test hiện tại dùng chung (chưa có augmentation)
    # Về sau (Phase 3) ta sẽ thêm augmentation vào train
    transforms = A.Compose([
        # Resize cạnh dài về image_size, giữ nguyên aspect ratio
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_CUBIC),

        # Pad về đúng image_size x image_size (thêm viền đen vào cạnh ngắn)
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
        ),

        # Chuẩn hóa theo ImageNet stats
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # Chuyển numpy array (H,W,C) -> tensor (C,H,W)
        ToTensorV2(),
    ])

    return transforms