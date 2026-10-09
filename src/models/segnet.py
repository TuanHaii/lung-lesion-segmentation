
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, init_features: int = 32):
        super(SegNet, self).__init__()
        
        # --- ENCODER ---
        # Stage 1
        self.enc_conv1_1 = nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1)
        self.enc_bn1_1 = nn.BatchNorm2d(init_features)
        self.enc_conv1_2 = nn.Conv2d(init_features, init_features, kernel_size=3, padding=1)
        self.enc_bn1_2 = nn.BatchNorm2d(init_features)

        # Stage 2
        self.enc_conv2_1 = nn.Conv2d(init_features, init_features * 2, kernel_size=3, padding=1)
        self.enc_bn2_1 = nn.BatchNorm2d(init_features * 2)
        self.enc_conv2_2 = nn.Conv2d(init_features * 2, init_features * 2, kernel_size=3, padding=1)
        self.enc_bn2_2 = nn.BatchNorm2d(init_features * 2)

        # Stage 3
        self.enc_conv3_1 = nn.Conv2d(init_features * 2, init_features * 4, kernel_size=3, padding=1)
        self.enc_bn3_1 = nn.BatchNorm2d(init_features * 4)
        self.enc_conv3_2 = nn.Conv2d(init_features * 4, init_features * 4, kernel_size=3, padding=1)
        self.enc_bn3_2 = nn.BatchNorm2d(init_features * 4)

        # Stage 4
        self.enc_conv4_1 = nn.Conv2d(init_features * 4, init_features * 8, kernel_size=3, padding=1)
        self.enc_bn4_1 = nn.BatchNorm2d(init_features * 8)
        self.enc_conv4_2 = nn.Conv2d(init_features * 8, init_features * 8, kernel_size=3, padding=1)
        self.enc_bn4_2 = nn.BatchNorm2d(init_features * 8)

        # --- DECODER (Max-Unpooling) ---
        # Stage 4
        self.dec_conv4_2 = nn.Conv2d(init_features * 8, init_features * 8, kernel_size=3, padding=1)
        self.dec_bn4_2 = nn.BatchNorm2d(init_features * 8)
        self.dec_conv4_1 = nn.Conv2d(init_features * 8, init_features * 4, kernel_size=3, padding=1)
        self.dec_bn4_1 = nn.BatchNorm2d(init_features * 4)

        # Stage 3
        self.dec_conv3_2 = nn.Conv2d(init_features * 4, init_features * 4, kernel_size=3, padding=1)
        self.dec_bn3_2 = nn.BatchNorm2d(init_features * 4)
        self.dec_conv3_1 = nn.Conv2d(init_features * 4, init_features * 2, kernel_size=3, padding=1)
        self.dec_bn3_1 = nn.BatchNorm2d(init_features * 2)

        # Stage 2
        self.dec_conv2_2 = nn.Conv2d(init_features * 2, init_features * 2, kernel_size=3, padding=1)
        self.dec_bn2_2 = nn.BatchNorm2d(init_features * 2)
        self.dec_conv2_1 = nn.Conv2d(init_features * 2, init_features, kernel_size=3, padding=1)
        self.dec_bn2_1 = nn.BatchNorm2d(init_features)

        # Stage 1
        self.dec_conv1_2 = nn.Conv2d(init_features, init_features, kernel_size=3, padding=1)
        self.dec_bn1_2 = nn.BatchNorm2d(init_features)
        self.dec_conv1_1 = nn.Conv2d(init_features, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encode 1
        x = F.relu(self.enc_bn1_1(self.enc_conv1_1(x)))
        x = F.relu(self.enc_bn1_2(self.enc_conv1_2(x)))
        s1 = x.size()
        x, idx1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encode 2
        x = F.relu(self.enc_bn2_1(self.enc_conv2_1(x)))
        x = F.relu(self.enc_bn2_2(self.enc_conv2_2(x)))
        s2 = x.size()
        x, idx2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encode 3
        x = F.relu(self.enc_bn3_1(self.enc_conv3_1(x)))
        x = F.relu(self.enc_bn3_2(self.enc_conv3_2(x)))
        s3 = x.size()
        x, idx3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encode 4
        x = F.relu(self.enc_bn4_1(self.enc_conv4_1(x)))
        x = F.relu(self.enc_bn4_2(self.enc_conv4_2(x)))
        s4 = x.size()
        x, idx4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Decode 4
        x = F.max_unpool2d(x, idx4, kernel_size=2, stride=2, output_size=s4)
        x = F.relu(self.dec_bn4_2(self.dec_conv4_2(x)))
        x = F.relu(self.dec_bn4_1(self.dec_conv4_1(x)))

        # Decode 3
        x = F.max_unpool2d(x, idx3, kernel_size=2, stride=2, output_size=s3)
        x = F.relu(self.dec_bn3_2(self.dec_conv3_2(x)))
        x = F.relu(self.dec_bn3_1(self.dec_conv3_1(x)))

        # Decode 2
        x = F.max_unpool2d(x, idx2, kernel_size=2, stride=2, output_size=s2)
        x = F.relu(self.dec_bn2_2(self.dec_conv2_2(x)))
        x = F.relu(self.dec_bn2_1(self.dec_conv2_1(x)))

        # Decode 1
        x = F.max_unpool2d(x, idx1, kernel_size=2, stride=2, output_size=s1)
        x = F.relu(self.dec_bn1_2(self.dec_conv1_2(x)))
        out = self.dec_conv1_1(x)

        return out
