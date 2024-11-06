import torch
from torch import nn, clip, tensor, Tensor
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
from torchvision import transforms
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL.Image import Image
from pathlib import Path
from abc import ABC, abstractmethod
import cv2
from typing import Tuple  


# データセット定義
class DataSetBase(data.Dataset, ABC):
    def __init__(self, image_path: Path):
        self.images = list(image_path.iterdir())
        self.max_num_sample = len(self.images)
        
    def __len__(self) -> int:
        return self.max_num_sample
    
    @abstractmethod
    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        pass
    
    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return image
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        image_path = self.images[index % len(self.images)]
        high_resolution_image = self.preprocess_high_resolution_image(PIL.Image.open(image_path))
        low_resolution_image = self.get_low_resolution_image(high_resolution_image, image_path)
        return transforms.ToTensor()(low_resolution_image), transforms.ToTensor()(high_resolution_image)

class TrainDataSet(DataSetBase):
    def __init__(self, image_path: Path, num_image_per_epoch: int = 2000):
        super().__init__(image_path)
        self.max_num_sample = num_image_per_epoch

    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        return transforms.Resize((image.size[0] // 4, image.size[1] // 4), transforms.InterpolationMode.BICUBIC)(image.copy())
    
    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return transforms.Compose([
            transforms.RandomCrop(size=512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])(image)

class ValidationDataSet(DataSetBase):
    def __init__(self, high_resolution_image_path: Path, low_resolution_image_path: Path):
        super().__init__(high_resolution_image_path)
        self.high_resolution_image_path = high_resolution_image_path
        self.low_resolution_image_path = low_resolution_image_path

    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        return PIL.Image.open(self.low_resolution_image_path / path.relative_to(self.high_resolution_image_path))

def get_dataset() -> Tuple[TrainDataSet, ValidationDataSet]:
    return TrainDataSet(Path("dataset/train"), 5000 * 60), ValidationDataSet(Path("dataset/validation/original"), Path("dataset/validation/0.25x"))

class ESPCN4x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv_1.bias)
        
        self.act = nn.ReLU()
        
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv_2.bias)
        
        self.conv_3 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv_3.bias)
        
        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_4.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv_4.bias)
        
        self.conv_5 = nn.Conv2d(in_channels=32, out_channels=(1 * self.scale * self.scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_5.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv_5.bias)
        
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, X_in: tensor) -> tensor:
        X = X_in.reshape(-1, 1, X_in.shape[-2], X_in.shape[-1])
        X = self.act(self.conv_1(X))
        X = self.act(self.conv_2(X))
        X = self.act(self.conv_3(X))
        X = self.act(self.conv_4(X))
        X = self.conv_5(X)
        X = self.pixel_shuffle(X)
        X = X.reshape(-1, 3, X.shape[-2], X.shape[-1])
        X_out = clip(X, 0.0, 1.0)
        return X_out

# 学習パラメーター
batch_size = 64
num_workers = 16
num_epoch = 40
learning_rate = 1e-3
gpu_id =2

def train():
    to_image = transforms.ToPILImage()
    
    def calc_psnr(image1: Tensor, image2: Tensor):
        image1 = cv2.cvtColor((np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor((np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.PSNR(image1, image2)

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    model = ESPCN4x()
    model.to(device)

    train_dataset, validation_dataset = get_dataset()
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_data_loader = data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 65, 80, 90], gamma=0.7) 
    criterion = MSELoss()

    train_loss_history = []
    validation_loss_history = []
    train_psnr_history = []
    validation_psnr_history = []
    
    best_validation_psnr = 0.0  # 最良の検証PSNRを記録
    best_model_path = "model.onnx"  # 最良モデルの保存先

    for epoch in trange(num_epoch, desc="EPOCH"):
        try:
            # 学習
            model.train()
            train_loss = 0.0 
            validation_loss = 0.0 
            train_psnr = 0.0
            validation_psnr = 0.0

            for idx, (low_resolution_image, high_resolution_image) in tqdm(enumerate(train_data_loader), desc=f"EPOCH[{epoch}] TRAIN", total=len(train_data_loader)):
                low_resolution_image = low_resolution_image.to(device)
                high_resolution_image = high_resolution_image.to(device)
                optimizer.zero_grad()
                output = model(low_resolution_image)
                loss = criterion(output, high_resolution_image)
                loss.backward()
                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image):   
                    train_psnr += calc_psnr(image1, image2)
                optimizer.step()

            scheduler.step()

            train_loss /= len(train_dataset)
            train_psnr /= len(train_dataset)
            train_loss_history.append(train_loss)
            train_psnr_history.append(train_psnr)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train PSNR = {train_psnr:.4f}")

            # 検証
            model.eval()
            with torch.no_grad():
                for idx, (low_resolution_image, high_resolution_image) in tqdm(enumerate(validation_data_loader), desc=f"EPOCH[{epoch}] VALIDATION", total=len(validation_data_loader)):
                    low_resolution_image = low_resolution_image.to(device)
                    high_resolution_image = high_resolution_image.to(device)
                    output = model(low_resolution_image)
                    loss = criterion(output, high_resolution_image)
                    validation_loss += loss.item() * low_resolution_image.size(0)
                    for image1, image2 in zip(output, high_resolution_image):   
                        validation_psnr += calc_psnr(image1, image2)

            validation_loss /= len(validation_dataset)
            validation_psnr /= len(validation_dataset)
            validation_loss_history.append(validation_loss)
            validation_psnr_history.append(validation_psnr)
            print(f"Epoch {epoch+1}: Validation Loss = {validation_loss:.4f}, Validation PSNR = {validation_psnr:.4f}")

            # 最良の検証結果の保存
            if  epoch > 10 and validation_psnr > best_validation_psnr:
                best_validation_psnr = validation_psnr
                # モデルをONNX形式で保存
                model.to(torch.device("cpu"))
                dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
                torch.onnx.export(model, dummy_input, best_model_path, 
                                  opset_version=17,
                                  input_names=["input"],
                                  output_names=["output"],
                                  dynamic_axes={"input": {2: "height", 3:"width"}})
                model.to(device)  # 再度GPUに戻す

        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")

    # 結果をCSVファイルに保存
    history_df = pd.DataFrame({
        'Epoch': range(1, num_epoch + 1),
        'Train Loss': train_loss_history,
        'Validation Loss': validation_loss_history,
        'Train PSNR': train_psnr_history,
        'Validation PSNR': validation_psnr_history
    })
    history_df.to_csv("training_history.csv", index=False)

    # LossとPSNRを可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epoch + 1), train_loss_history, marker='o', label='Train Loss')
    plt.plot(range(1, num_epoch + 1), validation_loss_history, marker='o', label='Validation Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epoch + 1), train_psnr_history, marker='o', label='Train PSNR')
    plt.plot(range(1, num_epoch + 1), validation_psnr_history, marker='o', label='Validation PSNR')
    plt.title('Train and Validation PSNR Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

    # 最後に最良モデルのパスを表示
    print(f"Best model saved as: {best_model_path}")

if __name__ == "__main__":
    train()
