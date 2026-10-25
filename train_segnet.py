"""
Task ID: EXP-02 / MODEL-06 (Sprint W3 - Baseline Training)
Mô tả: Huấn luyện mô hình SegNet với Hybrid Loss (BCE + Dice Loss)
Đảm bảo điều kiện thực nghiệm công bằng (Fair Comparison) với U-Net và Attention U-Net.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.segnet import SegNet
from src.losses.hybrid_loss import HybridLoss
from src.data.dataset import get_dataloader
from src.utils.metrics import compute_metrics

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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks, _ in tqdm(dataloader, desc="Training SegNet", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    for images, masks, _ in tqdm(dataloader, desc="Validating SegNet", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        outputs = model(images)
        loss = criterion(outputs, masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        metrics = compute_metrics(outputs, masks, threshold=0.5)
        total_dice += metrics["dice"] * batch_size
        total_iou += metrics["iou"] * batch_size
        total_samples += batch_size

    return {
        "loss": running_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SegNet Baseline Model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw data directory")
    args = parser.parse_args()


    logging.info("=== BẮT ĐẦU HUẤN LUYỆN SEGNet (Experiment ID: EXP-SEGNET-001) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo DataLoaders thực tế
    raw_data_dir = args.data_dir or os.environ.get("RAW_DATA_DIR")
    train_loader = get_dataloader(split="train", batch_size=args.batch_size, raw_data_dir=raw_data_dir)
    val_loader = get_dataloader(split="val", batch_size=args.batch_size, raw_data_dir=raw_data_dir)

    logging.info(f"Tập Train: {len(train_loader.dataset)} mẫu | Tập Val: {len(val_loader.dataset)} mẫu")

    # 2. Khởi tạo Mô hình, Loss function và Optimizer
    model = SegNet(in_channels=1, num_classes=1, init_features=32).to(device)
    criterion = HybridLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Vòng lặp huấn luyện mẫu (Training Loop)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = validate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        logging.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_res['loss']:.4f} | "
            f"Val Dice: {val_res['dice']:.4f} | "
            f"Val IoU: {val_res['iou']:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Lưu checkpoint tốt nhất dựa trên Val Dice
        if val_res["dice"] > best_dice:
            best_dice = val_res["dice"]
            checkpoint_path = "checkpoints/segnet_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_dice,
            }, checkpoint_path)
            logging.info(f"-> Đã cập nhật checkpoint tốt nhất tại: {checkpoint_path} (Dice: {best_dice:.4f})")

    logging.info(f"=== HOÀN TẤT HUẤN LUYỆN SEGNet (Best Dice: {best_dice:.4f}) ===")


if __name__ == "__main__":
    main()
