import torch
import torch.nn as nn

class StructureEncoder(nn.Module):
    """对汉字笔画和骨架进一步解析的网络"""
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, struct_feat):
        # struct_feat: [batch, in_dim]
        return self.net(struct_feat)  # [batch, out_dim]
