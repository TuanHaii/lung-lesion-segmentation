import torch
import torch.nn as nn
from src.losses.bce_loss import BCELoss # hoặc nn.BCEWithLogitsLoss
from src.losses.dice_loss import DiceLoss

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + self.beta * dice_loss