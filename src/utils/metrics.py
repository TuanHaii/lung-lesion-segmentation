import torch

def compute_metrics(pred_mask, true_mask, threshold=0.5, smooth=1e-6):
    """
    pred_mask: logits tensor [B, 1, H, W]
    true_mask: binary ground truth tensor [B, 1, H, W]
    """
    # The training/validation loop always passes raw logits.  Always apply
    # sigmoid here; the old value-based heuristic silently skipped it while
    # logits were still small, which made early-epoch validation metrics wrong.
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