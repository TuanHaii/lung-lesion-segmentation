import torch
import torch.nn as nn


class BCELossWrapper(nn.Module):
    """
    Task MODEL-03: Binary Cross-Entropy Loss nhận Logits đầu vào.
    Sử dụng nn.BCEWithLogitsLoss để đảm bảo độ ổn định số học.
    """
    def __init__(self, pos_weight: float | None = None):
        super(BCELossWrapper, self).__init__()
        if pos_weight is not None:
            weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Tensor đầu ra chưa qua Sigmoid [B, 1, H, W]
            targets (torch.Tensor): Nhãn Ground Truth {0, 1} [B, 1, H, W]
        """
        return self.criterion(logits, targets.float())