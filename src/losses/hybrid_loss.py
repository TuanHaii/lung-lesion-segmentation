import torch
import torch.nn as nn
# Import SoftDiceLoss từ Task MODEL-02
from .dice_loss import SoftDiceLoss


class HybridLoss(nn.Module):
    """
    Task MODEL-04: Cài đặt Hybrid Loss Function (BCE + Soft Dice Loss).
    Hàm mục tiêu chuẩn: L_hybrid = alpha * L_BCE + beta * L_Dice
    Mặc định: alpha = 0.5, beta = 0.5 theo đặc tả đề tài.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-5):
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Đầu ra thô chưa kích hoạt từ mô hình [B, 1, H, W]
            targets (torch.Tensor): Ground Truth nhị phân {0, 1} [B, 1, H, W]
        Returns:
            torch.Tensor: Giá trị tổn thất kết hợp (Scalar)
        """
        # Bắt buộc ép kiểu targets sang float32 để tránh lỗi runtime với BCEWithLogitsLoss
        targets = targets.float()
        
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss