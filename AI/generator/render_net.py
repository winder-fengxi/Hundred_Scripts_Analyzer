import torch
import torch.nn as nn
from generator.svg_render import VectorRenderer  # 矢量渲染模块
from generator.modeling import UpBlock, ACmix      # 注意卷积与上采样模块
from utils import apply_diffusion         # 墨迹扩散模拟

class CalligraphyGenerator(nn.Module):
    """
    书法图像生成器：接收融合特征，输出最终带墨迹效果的图像
    """
    def __init__(self, in_dim=256, mid_dim=128, out_channels=1):
        super().__init__()
        # 上采样 + 注意力卷积 + 矢量渲染
        self.render_layers = nn.Sequential(
            UpBlock(in_dim, mid_dim),  # 上采样并减少通道
            ACmix(mid_dim),             # 注意力卷积层
            VectorRenderer()            # SVG 矢量转位图
        )

    def forward(self, fused_feat):
        # fused_feat: [B, in_dim]
        # 扩展到特征图 [B, in_dim, H, W]，这里假设 H=W=8
        B = fused_feat.size(0)
        feat_map = fused_feat.view(B, -1, 1, 1).expand(-1, -1, 8, 8)
        # 渲染初始位图
        bitmap = self.render_layers(feat_map)  # [B, C, H', W']
        # 添加墨迹扩散效果
        output = self._add_ink_effect(bitmap)
        return output

    def _add_ink_effect(self, img):
        """
        模拟墨迹在纸上扩散效果
        """
        # apply_diffusion: 对 img 进行扩散迭代
        return apply_diffusion(img, steps=5)
