import torch
import torch.nn as nn
from .dice_loss import DiceLoss

class HybridLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss