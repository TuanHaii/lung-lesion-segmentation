"""
Task ID: EXP-02 / MODEL-06 (Sprint W3 - Baseline Training)
Mô tả: Huấn luyện mô hình SegNet với Hybrid Loss (BCE + Dice Loss)
Đảm bảo điều kiện thực nghiệm công bằng (Fair Comparison) với U-Net.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.segnet import SegNet

# --- Cấu hình Logging ---
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

logging.basicConfig(
    filename="logs/segnet_train.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1.0 - dice

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.alpha * bce_loss + self.beta * dice_loss

def main():
    logging.info("=== BẮT ĐẦU HUẤN LUYỆN SEGNet (Experiment ID: EXP-SEGNET-001) ===")
    
    # 1. Thiết bị huấn luyện (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Sử dụng thiết bị: {device}")

    # 2. Khởi tạo Dummy Dataset mô phỏng (Thay thế bằng DataLoader thực tế từ Đông khi sẵn sàng)
    torch.manual_seed(42)
    dummy_inputs = torch.randn(16, 1, 256, 256)
    dummy_masks = torch.randint(0, 2, (16, 1, 256, 256)).float()
    
    dataset = TensorDataset(dummy_inputs, dummy_masks)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. Khởi tạo Mô hình, Loss function và Optimizer
    model = SegNet(in_channels=1, num_classes=1).to(device)
    criterion = HybridLoss(alpha=0.5, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. Vòng lặp huấn luyện mẫu (Training Loop)
    epochs = 5
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

        # Lưu checkpoint tốt nhất
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = "checkpoints/segnet_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"-> Đã cập nhật checkpoint tốt nhất tại: {checkpoint_path}")

    logging.info("=== HOÀN TẤT HUẤN LUYỆN SEGNet ===")

if __name__ == "__main__":
    main()
