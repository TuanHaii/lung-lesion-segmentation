import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate Module (Oktay et al., 2018):
    - x: Skip connection tensor từ Encoder (đặc trưng không gian mức thấp)
    - g: Gating signal từ Decoder (đặc trưng ngữ nghĩa mức cao)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # Tích chập 1x1 cho tín hiệu điều khiển g từ Decoder
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Tích chập 1x1 cho skip connection x từ Encoder
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Tính toán ma trận trọng số chú ý (Attention Map)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Chiếu g và x về cùng không gian kênh F_int
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Cộng đặc trưng và tính bản đồ trọng số chú ý alpha (giá trị trong khoảng [0, 1])
        net = self.relu(g1 + x1)
        alpha = self.psi(net)

        # Nhân trọng số chú ý alpha với tensor x ban đầu để lọc nhiễu nền
        return x * alpha


if __name__ == "__main__":
    # Sanity check AttentionGate
    g = torch.randn(2, 64, 32, 32)
    x = torch.randn(2, 64, 32, 32)
    ag = AttentionGate(F_g=64, F_l=64, F_int=32)
    out = ag(g, x)
    assert out.shape == x.shape, "AttentionGate output shape mismatch!"
    print(">> ATTENTION GATE SANITY CHECK PASSED <<")
