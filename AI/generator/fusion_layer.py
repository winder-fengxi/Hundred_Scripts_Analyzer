import torch
import torch.nn as nn

class StyleFusion(nn.Module):
    """
    跨模态特征融合层：融合作者风格向量与用户内容结构特征
    """
    def __init__(self, style_dim=512, content_dim=256, fuse_dim=256):
        super().__init__()
        # 投影层，将风格和内容映射到相同维度
        self.style_proj = nn.Linear(style_dim, fuse_dim)
        self.content_proj = nn.Linear(content_dim, fuse_dim)
        # 可学习门控参数，用于动态融合
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, style_vec, content_vec):
        # style_vec: [B, style_dim]
        # content_vec: [B, content_dim]
        style_p = self.style_proj(style_vec)      # [B, fuse_dim]
        content_p = self.content_proj(content_vec)  # [B, fuse_dim]
        # 门控融合：sigmoid(gamma * style_p + content_p)
        gate = torch.sigmoid(self.gamma * style_p + content_p)
        fused = gate * style_p + (1 - gate) * content_p
        return fused  # [B, fuse_dim]
