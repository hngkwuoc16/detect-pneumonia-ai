import sys
sys.path.append('d:/detect-pneumonia-ai')  # để import được src

from src.data.datamodule import ChestXrayDataModule

def main():
    # Đường dẫn đến splits.csv (bạn cần điều chỉnh cho máy local)
    csv_path = 'd:/detect-pneumonia-ai/data/splits.csv'  # sửa lại cho đúng

    dm = ChestXrayDataModule(csv_path=csv_path, batch_size=8, num_workers=0) #vì đây là module test thôi nên num_workers=0 để tránh lỗi đa luồng, batch size = 8 để test nhanh.
    dm.setup()  #gọi setup trước khi gọi train_dataloader() để đọc CSV và tạo transforms

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch

    print("Batch images shape:", images.shape)  # (8, 3, 224, 224)
    print("Batch labels shape:", labels.shape)  # (8,)
    print("Labels:", labels)

if __name__ == "__main__":
    main()