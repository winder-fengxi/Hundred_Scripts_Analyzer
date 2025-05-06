
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, stride=1):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expansion
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEModule(out_channels)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.se(self.block(x))
        else:
            return self.se(self.block(x))

class EfficientNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetSmall, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            MBConv(32, 16, stride=1),
            MBConv(16, 24, stride=2),
            MBConv(24, 24, stride=1),
            MBConv(24, 40, stride=2),
            MBConv(40, 80, stride=2),
            MBConv(80, 112, stride=1),
            MBConv(112, 192, stride=2),
            MBConv(192, 320, stride=1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
