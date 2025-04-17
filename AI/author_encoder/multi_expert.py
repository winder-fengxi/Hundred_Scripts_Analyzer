import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class MultiExpertEncoder(nn.Module):
    """基于多专家网络和交叉注意力的书法家风格编码器"""
    def __init__(self, font_num=5, style_dim=512):
        super().__init__()
        # 初始化多个预训练 EfficientNet 作为专家网络
        self.experts = nn.ModuleList([
            EfficientNet.from_pretrained('efficientnet-b3')
            for _ in range(font_num)
        ])
        # 交叉注意力层，融合所有专家的特征
        # EfficientNet-B3 输出通道 1536
        self.attention = nn.Sequential(
            nn.Linear(1536 * font_num, style_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [batch, C, H, W]
        # 从每个专家提取特征
        feats = [expert.extract_features(x) for expert in self.experts]
        # feats[i]: [batch, 1536, h, w]
        # 按照空间均值池化并拼接
        pooled = [f.mean(dim=[2,3]) for f in feats]  # 每个 -> [batch,1536]
        concat = torch.cat(pooled, dim=1)            # [batch,1536*font_num]
        # 交叉注意力变换
        style_vec = self.attention(concat)           # [batch, style_dim]
        return style_vec

