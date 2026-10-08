import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate Module:
    - x: Skip connection tensor từ Encoder (đặc trưng không gian mức thấp)
    - g: Gating signal từ Decoder (đặc trưng ngữ nghĩa mức cao)
    """
    def __init__(self, F_g, F_l, F_int):
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

    def forward(self, g, x):
        # Chiếu g và x về cùng không gian kênh F_int
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Cộng đặc trưng và tính bản đồ trọng số chú ý alpha (giá trị trong khoảng [0, 1])
        net = self.relu(g1 + x1)
        alpha = self.psi(net)
        
        # Nhân trọng số chú ý alpha với tensor x ban đầu để lọc nhiễu nền
        return x * alpha


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.ag = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. ENCODER (Contracting Path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 2. BOTTLENECK
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 3. DECODER + ATTENTION GATES (Expanding Path)
        for feature in reversed(features):
            # Upsampling bằng Transposed Convolution
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # Khởi tạo Attention Gate cho từng cấp độ skip connection
            self.ag.append(
                AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            # Khối tích chập sau khi ghép nối (Concat feature * 2 -> feature)
            self.ups.append(DoubleConv(feature * 2, feature))

        # 4. FINAL CLASSIFIER
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # --- Luồng xử lý ENCODER ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- Luồng xử lý BOTTLENECK ---
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # --- Luồng xử lý DECODER ---
        ag_idx = 0
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample tín hiệu từ tầng sâu
            
            skip_conn = skip_connections[ag_idx]
            
            # Lọc skip connection qua Attention Gate trước khi Concat
            ag_skip = self.ag[ag_idx](g=x, x=skip_conn)
            ag_idx += 1
            
            # Ghép đặc trưng đã lọc với tín hiệu đã upsample
            concat_x = torch.cat((ag_skip, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


if __name__ == "__main__":
    # Sanity check kích thước Tensor đầu vào / đầu ra
    dummy_input = torch.randn(2, 1, 256, 256)  # [Batch_size=2, Channel=1, H=256, W=256]
    model = AttentionUNet(in_channels=1, out_channels=1)
    output = model(dummy_input)
    
    print("Shape đầu vào:", dummy_input.shape)
    print("Shape đầu ra: ", output.shape)
    
    assert output.shape == dummy_input.shape, "Lỗi kích thước output không trùng khớp!"
    print(">> ATTENTION U-NET SANITY CHECK PASSED <<")