import torch
import torch.nn as nn
import torchvision
from utils import skeletonize  # 骨架化方法

class ContentEncoder(nn.Module):
    """用户内容特征提取：视觉和结构双路编码"""
    def __init__(self, pretrained=True, out_dim=256):
        super().__init__()
        # 视觉特征提取网络（ResNet-50 去掉最后两层）
        base = torchvision.models.resnet50(pretrained=pretrained)
        self.visual_extractor = nn.Sequential(*list(base.children())[:-2],
                                              nn.AdaptiveAvgPool2d(1))  # 输出 2048
        # 结构特征分支，将视觉特征映射到低维表示
        self.struct_fc = nn.Linear(2048, out_dim)

    def forward(self, img):
        # img: [batch, C, H, W]
        # 骨架化预处理
        skel = skeletonize(img)          # [batch, C, H, W]
        # 视觉特征
        vis_feat = self.visual_extractor(img).view(img.size(0), -1)        # [batch, 2048]
        # 结构特征
        struc_feat = self.visual_extractor(skel).view(img.size(0), -1)     # [batch, 2048]
        struc_feat = self.struct_fc(struc_feat)                           # [batch, out_dim]
        return {
            'visual': vis_feat,
            'structure': struc_feat
        }
