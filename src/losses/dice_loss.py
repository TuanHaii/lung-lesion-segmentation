import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """
    Task MODEL-02: Soft Dice Loss phục vụ bài toán phân đoạn tổn thương CT phổi.
    
    Công thức:
        L_Dice = 1 - (2 * sum(p * g) + eps) / (sum(p^2) + sum(g^2) + eps)
    Trong đó:
        p: Xác suất sau hàm Sigmoid
        g: Ground Truth nhị phân {0, 1}
        eps: 1e-5 (smooth factor chống chia cho 0 và ổn định gradient)
    """
    def __init__(self, smooth: float = 1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện tính toán Soft Dice Loss trên từng lát cắt (Slice-level).

        Tham số:
            logits (torch.Tensor): Tensor đầu ra chưa kích hoạt từ mô hình [B, 1, H, W]
            targets (torch.Tensor): Nhãn Ground Truth nhị phân [B, 1, H, W]

        Trả về:
            torch.Tensor: Giá trị tổn thất trung bình trên toàn bộ Batch (Scalar)
        """
        # 1. Kích hoạt sigmoid đưa logits về khoảng xác suất [0, 1]
        probs = torch.sigmoid(logits)

        # 2. Đảm bảo nhãn targets cùng kiểu dữ liệu float với probs
        targets = targets.float()

        # 3. Làm phẳng tensor theo từng lát cắt: [B, C, H, W] -> [B, C * H * W]
        batch_size = probs.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # 4. Tính toán phần giao (Intersection) và tổng bình phương (Cardinality)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = (probs_flat ** 2).sum(dim=1) + (targets_flat ** 2).sum(dim=1)

        # 5. Tính chỉ số Soft Dice và hàm Loss cho từng mẫu trong batch
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice_score

        # 6. Trả về giá trị trung bình trên toàn bộ batch (Slice-level Macro Average)
        return loss.mean()


if __name__ == "__main__":
    # Test nhanh cấu hình khởi tạo
    loss_fn = SoftDiceLoss(smooth=1e-5)
    print(f"SoftDiceLoss initialized with smooth={loss_fn.smooth}")