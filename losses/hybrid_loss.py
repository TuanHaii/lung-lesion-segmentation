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
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + self.beta * dice_loss