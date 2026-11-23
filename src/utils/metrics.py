import torch
from typing import Dict, Tuple


def compute_metrics(pred_mask, true_mask, threshold=0.5, smooth=1e-6):
    """
    Legacy metric computation (batch-flattened micro metric).
    pred_mask: logits tensor [B, 1, H, W]
    true_mask: binary ground truth tensor [B, 1, H, W]
    """
    probs = torch.sigmoid(pred_mask)
    preds = (probs > threshold).float()
    
    preds = preds.view(-1)
    true_mask = true_mask.view(-1)
    
    tp = (preds * true_mask).sum()
    fp = (preds * (1 - true_mask)).sum()
    fn = ((1 - preds) * true_mask).sum()
    tn = ((1 - preds) * (1 - true_mask)).sum()
    
    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item()
    }


def compute_sample_metrics(
    pred_logits: torch.Tensor,
    true_mask: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Compute true per-sample macro metrics averaged over the batch.

    Properly handles empty ground truth masks:
    - If both ground truth and prediction are empty: Dice=1.0, IoU=1.0, Precision=1.0, Recall=1.0
    - If ground truth is empty but prediction is non-empty: Dice=0.0, IoU=0.0, Precision=0.0, Recall=1.0

    Parameters
    ----------
    pred_logits : torch.Tensor
        Logits of shape [B, 1, H, W] or [B, H, W].
    true_mask : torch.Tensor
        Binary masks of shape [B, 1, H, W] or [B, H, W] in {0, 1}.

    Returns
    -------
    Dict[str, float]
        Averaged macro metrics over the B samples.
    """
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()

    if preds.ndim == 4 and preds.size(1) == 1:
        preds = preds.squeeze(1)
    if true_mask.ndim == 4 and true_mask.size(1) == 1:
        true_mask = true_mask.squeeze(1)

    batch_size = preds.size(0)
    dice_list = []
    iou_list = []
    prec_list = []
    rec_list = []

    for b in range(batch_size):
        p_flat = preds[b].reshape(-1)
        t_flat = true_mask[b].reshape(-1)

        tp = (p_flat * t_flat).sum().item()
        fp = (p_flat * (1.0 - t_flat)).sum().item()
        fn = ((1.0 - p_flat) * t_flat).sum().item()

        gt_pos = tp + fn
        pred_pos = tp + fp

        # Handle empty ground-truth cases
        if gt_pos == 0:
            if pred_pos == 0:
                dice = 1.0
                iou = 1.0
                precision = 1.0
                recall = 1.0
            else:
                dice = 0.0
                iou = 0.0
                precision = 0.0
                recall = 1.0
        else:
            dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
            iou = (tp + smooth) / (tp + fp + fn + smooth)
            precision = (tp + smooth) / (pred_pos + smooth)
            recall = (tp + smooth) / (gt_pos + smooth)

        dice_list.append(dice)
        iou_list.append(iou)
        prec_list.append(precision)
        rec_list.append(recall)

    return {
        "dice": float(sum(dice_list) / batch_size),
        "iou": float(sum(iou_list) / batch_size),
        "precision": float(sum(prec_list) / batch_size),
        "recall": float(sum(rec_list) / batch_size),
    }
