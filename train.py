import os
import sys
import yaml
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Thêm thư mục gốc của project vào sys.path để Python nhận diện module src
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.models.segnet import SegNet
from src.losses.hybrid_loss import HybridLoss
from src.data.dataset import get_dataloader
from src.utils.metrics import compute_metrics


def build_model(model_cfg: dict) -> nn.Module:
    """Khởi tạo mô hình dựa trên cấu hình YAML."""
    model_name = model_cfg.get("name", "AttentionUNet")
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", 1)

    if model_name in ["AttentionUNet", "AttentionUnet", "attention_unet"]:
        features = model_cfg.get("features", [64, 128, 256, 512])
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels, features=features)
    elif model_name in ["UNet", "Unet", "unet"]:
        features = model_cfg.get("features", [64, 128, 256, 512])
        return UNet(in_channels=in_channels, out_channels=out_channels, features=features)
    elif model_name in ["SegNet", "Segnet", "segnet"]:
        init_features = model_cfg.get("init_features", 32)
        return SegNet(in_channels=in_channels, num_classes=out_channels, init_features=init_features)
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}. Lựa chọn: AttentionUNet, UNet, SegNet.")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks, _ in tqdm(dataloader, desc="Training", leave=False):
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
    total_precision = 0.0
    total_recall = 0.0
    total_samples = 0

    for images, masks, _ in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        outputs = model(images)
        loss = criterion(outputs, masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        metrics = compute_metrics(outputs, masks, threshold=0.5)
        total_dice += metrics["dice"] * batch_size
        total_iou += metrics["iou"] * batch_size
        total_precision += metrics["precision"] * batch_size
        total_recall += metrics["recall"] * batch_size
        total_samples += batch_size

    val_loss = running_loss / total_samples
    val_dice = total_dice / total_samples
    val_iou = total_iou / total_samples
    val_precision = total_precision / total_samples
    val_recall = total_recall / total_samples

    return {
        "loss": val_loss,
        "dice": val_dice,
        "iou": val_iou,
        "precision": val_precision,
        "recall": val_recall,
    }


def main():
    parser = argparse.ArgumentParser(description="CT Lung Lesion Segmentation Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw data directory (for Kaggle)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda or cpu)")
    args = parser.parse_args()


    # 1. Load cấu hình YAML
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file config tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Xử lý ghi đè cấu hình từ arguments
    exp_id = config.get("experiment", {}).get("id") or config.get("experiment_name", "EXP-001")
    epochs = args.epochs or config.get("training", {}).get("epochs") or config.get("train", {}).get("epochs", 50)
    batch_size = args.batch_size or config.get("data", {}).get("batch_size") or config.get("train", {}).get("batch_size", 16)
    lr = args.lr or config.get("training", {}).get("optimizer", {}).get("lr") or config.get("train", {}).get("lr", 1e-4)
    raw_data_dir = args.data_dir or os.environ.get("RAW_DATA_DIR") or os.environ.get("KAGGLE_DATA_DIR")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f">> Starting Experiment: {exp_id}")
    print(f">> Device              : {device}")
    print(f">> Epochs              : {epochs} | Batch Size: {batch_size} | LR: {lr}")
    if raw_data_dir:
        print(f">> Raw Data Dir        : {raw_data_dir}")
    print("=" * 70)

    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "./checkpoints")
    log_dir = config.get("paths", {}).get("log_dir", "./logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 3. Khởi tạo DataLoaders
    num_workers = config.get("data", {}).get("num_workers", 2)
    pin_memory = config.get("data", {}).get("pin_memory", True)

    print(">> Initializing Train and Validation DataLoaders...")
    train_loader = get_dataloader(
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        raw_data_dir=raw_data_dir,
    )
    val_loader = get_dataloader(
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        raw_data_dir=raw_data_dir,
    )
    print(f"   [+] Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"   [+] Val samples  : {len(val_loader.dataset)} ({len(val_loader)} batches)")

    # 4. Khởi tạo Mô hình, Loss, Optimizer, Scheduler
    model = build_model(config.get("model", {})).to(device)

    # Cấu hình Loss
    loss_cfg = config.get("training", {}).get("loss", {})
    alpha_bce = loss_cfg.get("alpha_bce", 0.5)
    beta_dice = loss_cfg.get("beta_dice", 0.5)
    smooth_eps = loss_cfg.get("smooth_epsilon", 1e-5)
    criterion = HybridLoss(bce_weight=alpha_bce, dice_weight=beta_dice, smooth=smooth_eps).to(device)

    # Optimizer
    opt_cfg = config.get("training", {}).get("optimizer", {})
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    sched_cfg = config.get("training", {}).get("scheduler", {})
    t_max = sched_cfg.get("T_max", epochs)
    eta_min = sched_cfg.get("eta_min", 1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    best_val_dice = 0.0
    patience = config.get("training", {}).get("early_stopping_patience", 15)
    patience_counter = 0

    log_file_path = os.path.join(log_dir, f"{exp_id}_training.log")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_dice,val_iou,val_precision,val_recall,lr,elapsed_s\n")

    print("\n>> Entering Training Loop...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_metrics['loss']:.6f},{val_metrics['dice']:.6f},"
                f"{val_metrics['iou']:.6f},{val_metrics['precision']:.6f},{val_metrics['recall']:.6f},"
                f"{current_lr:.8f},{elapsed:.2f}\n"
            )

        # Lưu checkpoint tốt nhất theo Val Dice
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            patience_counter = 0
            best_save_path = os.path.join(checkpoint_dir, f"{exp_id}_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                best_save_path,
            )
            print(f"   [+] New best checkpoint saved to: {best_save_path} (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1

        # Luôn lưu last checkpoint để resume nếu bị ngắt
        last_save_path = os.path.join(checkpoint_dir, f"{exp_id}_last.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "last_val_dice": val_metrics["dice"],
                "config": config,
            },
            last_save_path,
        )

        if patience_counter >= patience:
            print(f"\n>> Early stopping triggered after {patience} epochs without improvement!")
            break

    print(f"\n>> Training Finished! Best Validation Dice: {best_val_dice:.4f}")
    print(f">> Checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    main()