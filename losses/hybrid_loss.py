import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.eps) / (probs.sum() + targets.sum() + self.eps)
        return 1.0 - dice

class HybridLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float | None = None,
        beta: float | None = None,
        smooth: float = 1e-5,
        eps: float | None = None,
    ):
        super().__init__()
        self.bce_weight = alpha if alpha is not None else bce_weight
        self.dice_weight = beta if beta is not None else dice_weight
        self.alpha = self.bce_weight
        self.beta = self.dice_weight

        smooth_val = eps if eps is not None else smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=smooth_val)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss