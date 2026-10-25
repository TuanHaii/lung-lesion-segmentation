import torch
import torch.nn as nn
from src.losses.bce_loss import BCELoss # hoặc nn.BCEWithLogitsLoss
from src.losses.dice_loss import DiceLoss

class HybridLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float | None = None,
        beta: float | None = None,
        smooth: float = 1e-5,
        eps: float | None = None,
        pos_weight: float | None = None,
    ):
        super(HybridLoss, self).__init__()
        # Allow alpha/beta as aliases for bce_weight/dice_weight
        self.bce_weight = alpha if alpha is not None else bce_weight
        self.dice_weight = beta if beta is not None else dice_weight
        self.alpha = self.bce_weight
        self.beta = self.dice_weight

        # Allow eps as alias for smooth
        self.smooth = eps if eps is not None else smooth

        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=self.smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss