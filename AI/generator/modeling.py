import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    """
    上采样模块：先双线性插值放大特征图，然后使用 1x1 卷积降维
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        # 用于降维或调整通道数
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 双线性插值上采样
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # 通道匹配
        out = self.conv1x1(x_up)
        return out

class ACmix(nn.Module):
    """
    注意力卷积模块：结合 空间自注意力 和 卷积 输出
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        # 标准卷积分支
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # 多头自注意力分支
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 卷积分支
        conv_out = self.conv(x)

        # 注意力分支
        # 重塑以符合 MultiheadAttention 输入 (seq_len, batch, embed_dim)
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)  # [N, B, C]
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        # 恢复形状
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)

        # 融合分支
        out = conv_out + attn_out
        # 可选归一化
        # 对通道维度做 LayerNorm（先permute再norm再还原）
        out_flat = out.permute(0,2,3,1)  # [B,H,W,C]
        out_norm = self.norm(out_flat).permute(0,3,1,2)
        return out_norm
