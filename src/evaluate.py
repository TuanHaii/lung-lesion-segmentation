"""
Task ID: EVAL-03 to EVAL-06 (Sprint W5 - Comprehensive Evaluation)
Mô tả: Script đánh giá mô hình phân đoạn tổn thương CT phổi trên tập Test độc lập.
Tuân thủ đặc tả nghiên cứu trong docs/05_OPERATIONAL_MANUAL.md:
- Slice-level Macro Average (Macro Average theo slice)
- Patient-level / Volume-level Aggregation (Tổng hợp TP, FP, FN theo ID bệnh nhân)
- Dual-reporting strategy: Báo cáo Lesion-containing Slices (GT > 0) và All Slices
- Xuất 2 bảng CSV: test_metrics_slice_level.csv & test_metrics_patient_level.csv
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.models.segnet import SegNet
from src.data.dataset import get_dataloader


def load_model_from_checkpoint(checkpoint_path: str, model_name: str, device: torch.device) -> nn.Module:
    """Tải trọng số mô hình từ file checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    if model_name in ["AttentionUNet", "AttentionUnet", "attention_unet"]:
        model = AttentionUNet(in_channels=1, out_channels=1)
    elif model_name in ["UNet", "Unet", "unet"]:
        model = UNet(in_channels=1, out_channels=1)
    elif model_name in ["SegNet", "Segnet", "segnet"]:
        model = SegNet(in_channels=1, num_classes=1, init_features=32)
    else:
        raise ValueError(f"Tên mô hình không hợp lệ: {model_name}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_slice_confusion(pred_binary: np.ndarray, target_binary: np.ndarray) -> Dict[str, int]:
    """Tính toán True Positive, False Positive, False Negative, True Negative trên lát cắt."""
    pred_flat = pred_binary.flatten().astype(bool)
    target_flat = target_binary.flatten().astype(bool)

    tp = int(np.logical_and(pred_flat, target_flat).sum())
    fp = int(np.logical_and(pred_flat, np.logical_not(target_flat)).sum())
    fn = int(np.logical_and(np.logical_not(pred_flat), target_flat).sum())
    tn = int(np.logical_and(np.logical_not(pred_flat), np.logical_not(target_flat)).sum())

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics_from_counts(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-6) -> Dict[str, float]:
    """Quy chuẩn xử lý Edge Case theo Mục B.6 trong 05_OPERATIONAL_MANUAL.md:
    1. GT rỗng, Pred rỗng -> TP=FP=FN=0 -> DSC=1.0, IoU=1.0, Precision=1.0, Recall=1.0
    2. GT rỗng, Pred có vết -> TP=0, FP>0 -> DSC=0.0, IoU=0.0, Precision=0.0, Recall=1.0
    3. GT có bệnh, Pred rỗng -> TP=0, FN>0 -> DSC=0.0, IoU=0.0, Precision=0.0, Recall=0.0
    """
    is_gt_empty = (tp + fn) == 0
    is_pred_empty = (tp + fp) == 0

    if is_gt_empty and is_pred_empty:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    elif is_gt_empty and not is_pred_empty:
        return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 1.0}
    elif not is_gt_empty and is_pred_empty:
        return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(np.clip(dice, 0.0, 1.0)),
        "iou": float(np.clip(iou, 0.0, 1.0)),
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
    }


def evaluate(
    model: nn.Module,
    test_loader,
    device: torch.device,
    output_dir: str = "reports",
    threshold: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    slice_records: List[Dict] = []
    patient_accumulators: Dict[str, Dict[str, int]] = {}

    print(">> Đang chạy suy luận trên Independent Test Set...")
    with torch.no_grad():
        for images, masks, metadata in tqdm(test_loader, desc="Testing", leave=True):
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()  # [B, H, W]
            masks_np = masks.squeeze(1).cpu().numpy()                # [B, H, W]

            batch_size = images.size(0)
            for i in range(batch_size):
                sample_id = metadata["sample_id"][i] if isinstance(metadata["sample_id"], list) else metadata["sample_id"]
                patient_id = metadata["patient_id"][i] if isinstance(metadata["patient_id"], list) else metadata["patient_id"]
                slice_index = int(metadata["slice_index"][i])
                has_lesion = int(metadata["lesion_present"][i])

                p_bin = (probs[i] > threshold).astype(np.uint8)
                t_bin = (masks_np[i] > 0.5).astype(np.uint8)

                counts = compute_confusion(p_bin, t_bin) if "compute_confusion" in globals() else compute_slice_confusion(p_bin, t_bin)
                metrics = compute_metrics_from_counts(counts["tp"], counts["fp"], counts["fn"], counts["tn"])

                slice_records.append({
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "slice_index": slice_index,
                    "lesion_present": has_lesion,
                    "tp": counts["tp"],
                    "fp": counts["fp"],
                    "fn": counts["fn"],
                    "tn": counts["tn"],
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                })

                # Tích luỹ thể tích cho từng bệnh nhân (Patient-level 3D aggregation)
                if patient_id not in patient_accumulators:
                    patient_accumulators[patient_id] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "slices": 0}
                patient_accumulators[patient_id]["tp"] += counts["tp"]
                patient_accumulators[patient_id]["fp"] += counts["fp"]
                patient_accumulators[patient_id]["fn"] += counts["fn"]
                patient_accumulators[patient_id]["tn"] += counts["tn"]
                patient_accumulators[patient_id]["slices"] += 1

    # 1. Xuất file test_metrics_slice_level.csv
    slice_csv_path = os.path.join(output_dir, "test_metrics_slice_level.csv")
    with open(slice_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "patient_id", "slice_index", "lesion_present",
            "tp", "fp", "fn", "tn", "dice", "iou", "precision", "recall"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(slice_records)

    # 2. Xuất file test_metrics_patient_level.csv
    patient_records = []
    for pid, counts in patient_accumulators.items():
        pm = compute_metrics_from_counts(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
        patient_records.append({
            "patient_id": pid,
            "total_slices": counts["slices"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "tn": counts["tn"],
            "patient_dice": pm["dice"],
            "patient_iou": pm["iou"],
            "patient_precision": pm["precision"],
            "patient_recall": pm["recall"],
        })

    patient_csv_path = os.path.join(output_dir, "test_metrics_patient_level.csv")
    with open(patient_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "patient_id", "total_slices", "tp", "fp", "fn", "tn",
            "patient_dice", "patient_iou", "patient_precision", "patient_recall"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patient_records)

    # 3. Tổng hợp Báo cáo theo Dual-Reporting Strategy (Mục B.6.2)
    all_dice = [r["dice"] for r in slice_records]
    all_iou = [r["iou"] for r in slice_records]
    all_prec = [r["precision"] for r in slice_records]
    all_rec = [r["recall"] for r in slice_records]

    lesion_slices = [r for r in slice_records if r["lesion_present"] == 1]
    lesion_dice = [r["dice"] for r in lesion_slices]
    lesion_iou = [r["iou"] for r in lesion_slices]
    lesion_prec = [r["precision"] for r in lesion_slices]
    lesion_rec = [r["recall"] for r in lesion_slices]

    print("\n" + "=" * 75)
    print(" BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (DUAL-REPORTING STRATEGY)")
    print("=" * 75)
    print(f">> CHẾ ĐỘ 1: LESION-CONTAINING SLICES [BÁO CÁO CHÍNH IN BÀI BÁO] (N={len(lesion_slices)})")
    if len(lesion_slices) > 0:
        print(f"   - Dice Score (DSC) : {np.mean(lesion_dice):.4f} +/- {np.std(lesion_dice):.4f}")
        print(f"   - IoU (Jaccard)    : {np.mean(lesion_iou):.4f} +/- {np.std(lesion_iou):.4f}")
        print(f"   - Precision (PPV)  : {np.mean(lesion_prec):.4f} +/- {np.std(lesion_prec):.4f}")
        print(f"   - Recall (TPR)     : {np.mean(lesion_rec):.4f} +/- {np.std(lesion_rec):.4f}")
    else:
        print("   - Không có lát cắt tổn thương nào trong tập kiểm thử.")

    print(f"\n>> CHẾ ĐỘ 2: ALL SLICES [BÁO CÁO BỔ TRỢ] (N={len(slice_records)})")
    print(f"   - Dice Score (DSC) : {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}")
    print(f"   - IoU (Jaccard)    : {np.mean(all_iou):.4f} +/- {np.std(all_iou):.4f}")
    print(f"   - Precision (PPV)  : {np.mean(all_prec):.4f} +/- {np.std(all_prec):.4f}")
    print(f"   - Recall (TPR)     : {np.mean(all_rec):.4f} +/- {np.std(all_rec):.4f}")

    pat_dice = [r["patient_dice"] for r in patient_records]
    print(f"\n>> CẤP ĐỘ BỆNH NHÂN (PATIENT-LEVEL VOLUME AGGREGATION) (N={len(patient_records)} patients)")
    print(f"   - Patient Mean DSC : {np.mean(pat_dice):.4f} +/- {np.std(pat_dice):.4f}")
    print("=" * 75)
    print(f"[+] Đã lưu bảng kết quả chi tiết:")
    print(f"    1. {slice_csv_path}")
    print(f"    2. {patient_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CT Lung Lesion Segmentation on Independent Test Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--model", type=str, default="AttentionUNet", choices=["AttentionUNet", "UNet", "SegNet"], help="Model architecture")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw data directory")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save evaluation reports")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binarization")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Thiết bị đánh giá: {device}")

    model = load_model_from_checkpoint(args.checkpoint, args.model, device)
    raw_data_dir = args.data_dir or os.environ.get("RAW_DATA_DIR")
    test_loader = get_dataloader(split="test", batch_size=1, shuffle=False, raw_data_dir=raw_data_dir)

    evaluate(model, test_loader, device, output_dir=args.output_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()
