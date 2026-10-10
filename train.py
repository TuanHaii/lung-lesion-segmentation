import os
import sys
import yaml
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Thêm thư mục gốc của project vào sys.path để Python nhận diện module src
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Sửa lại đường dẫn Import theo cấu trúc thư mục src/
from src.models.attention_unet import AttentionUNet
from src.losses.hybrid_loss import HybridLoss

def calculate_dice_score(logits, targets, eps=1e-5):
    """Tính toán Validation Dice Score không dùng gradient"""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)
    return dice.item()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
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
    running_dice = 0.0

    for images, masks in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        outputs = model(images)
        loss = criterion(outputs, masks)

        running_loss += loss.item() * images.size(0)
        running_dice += calculate_dice_score(outputs, masks) * images.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    val_dice = running_dice / len(dataloader.dataset)
    return val_loss, val_dice

def main(config_path="configs/config.yaml"):
    # 1. Load cấu hình
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Running Experiment: {config['experiment']['id']} on Device: {device}")

    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)

    # 2. Khởi tạo Mô hình, Loss, Optimizer, Scheduler
    model = AttentionUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)

    criterion = HybridLoss(
        bce_weight=config['training']['loss']['alpha_bce'],
        dice_weight=config['training']['loss']['beta_dice']
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['scheduler']['T_max'],
        eta_min=config['training']['scheduler']['eta_min']
    )

    # Note: Giả định train_loader và val_loader đã được xây dựng từ DataLoader (Đông phụ trách)
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)

    best_val_dice = 0.0
    patience_counter = 0

    print(">> Starting Training Loop...")
    for epoch in range(1, config['training']['epochs'] + 1):
        start_time = time.time()

        # Thực hiện 1 Epoch Train & Val
        # train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        # Cập nhật Learning Rate theo lịch trình Cosine
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - start_time
        
        # Giả lập log output minh họa
        # print(f"Epoch {epoch:03d}/{config['training']['epochs']} | "
        #       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        #       f"Val Dice: {val_dice:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.2f}s")

        # 3. Lưu best checkpoint dựa vào Val Dice
        # if val_dice > best_val_dice:
        #     best_val_dice = val_dice
        #     patience_counter = 0
        #     save_path = os.path.join(config['paths']['checkpoint_dir'], f"{config['experiment']['id']}_best.pth")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'best_val_dice': best_val_dice,
        #         'config': config
        #     }, save_path)
        #     print(f"   [+] Best checkpoint saved with Val Dice: {best_val_dice:.4f}")
        # else:
        #     patience_counter += 1

        # 4. Kiểm tra Early Stopping
        # if patience_counter >= config['training']['early_stopping_patience']:
        #     print(f">> Early stopping triggered at epoch {epoch}")
        #     break

if __name__ == "__main__":
    main()