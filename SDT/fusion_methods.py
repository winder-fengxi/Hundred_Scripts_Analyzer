# fusion_methods.py

import torch
import math


def slerp(alpha: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    球面线性插值 (Slerp)：
    在高维超球面上对向量 v0、v1（模长不限）做 α∈[0,1] 的线性插值。
    v0, v1: Tensor([B, D]) 或 Tensor([1, D])。
    返回同形状 Tensor。
    """
    # 1) 归一化
    v0_norm = torch.nn.functional.normalize(v0, p=2, dim=-1)
    v1_norm = torch.nn.functional.normalize(v1, p=2, dim=-1)

    # 2) 计算夹角 θ
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # [B,1] 或 [1,1]
    theta = torch.acos(dot)  # [B,1] 或 [1,1]

    # 3) sin(θ)
    sin_theta = torch.sin(theta)  # [B,1] 或 [1,1]
    near_zero = sin_theta.abs() < 1e-6  # 避免除零

    # 4) 计算权重
    w0 = torch.sin((1.0 - alpha) * theta) / sin_theta  # [B,1]
    w1 = torch.sin(alpha * theta) / sin_theta

    # 5) 如果 sin(θ) 太小，则 fallback 到线性插值
    w0 = torch.where(near_zero, (1.0 - alpha) * torch.ones_like(w0), w0)
    w1 = torch.where(near_zero, alpha * torch.ones_like(w1), w1)

    # 6) 插值
    return (w0 * v0) + (w1 * v1)


def geo_interp(alpha: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对数-指数插值（几何平均方式）：
      out[i] = (v0[i]+eps)^{α} * (v1[i]+eps)^{(1−α)}
    假设 v0, v1 中元素基本非负（或先添加 eps 保证正数）。
    v0, v1: Tensor([B, D]) 或 Tensor([1, D])。
    返回 Tensor([..., D])。
    """
    v0p = v0.clamp(min=eps)
    v1p = v1.clamp(min=eps)
    return (v0p ** alpha) * (v1p ** (1.0 - alpha))


def attention_fuse(char_emb: torch.Tensor,
                   mem_A: torch.Tensor,
                   mem_B: torch.Tensor) -> torch.Tensor:
    """
    基于注意力的融合示例。将作者 A/B 的 memory 张量按 batch 维拼接，
    然后用 char_emb 做 Query，计算注意力权重来加权聚合 A/B 的 memory。
    输入：
      char_emb: [B, 512]
      mem_A, mem_B: [4, B, 512]
    输出：
      mem_F: [4, B, 512] 融合后的 memory
    """
    # 拼接 A/B 内存 → [4, 2B, 512]
    mem_cat = torch.cat([mem_A, mem_B], dim=1)

    # 复制 char_emb 至时序长度 4 → [4, B, 512]
    char_4 = char_emb.unsqueeze(0).repeat(4, 1, 1)

    # 定义投影维度 d_k
    d_k = 64
    # 线性投影层：若在循环中频繁调用，可考虑将这两层模块提前定义并加载
    proj_q = torch.nn.Linear(512, d_k).to(char_emb.device)
    proj_k = torch.nn.Linear(512, d_k).to(char_emb.device)

    # 计算 Q, K
    Q = proj_q(char_4)          # [4, B, d_k]
    K = proj_k(mem_cat)         # [4, 2B, d_k]

    # 计算注意力打分，除以 sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)  # [4, B, 2B]
    attn = torch.softmax(scores, dim=-1)                            # [4, B, 2B]

    # 用注意力权重对 mem_cat 加权，得到 [4, B, 512]
    mem_F = torch.matmul(attn, mem_cat)

    return mem_F


class FusionMLP(torch.nn.Module):
    """
    可训练的融合网络示例，将 A/B 两路向量融合：
      out = g * A + (1−g) * B + offset
    其中 g = sigmoid(Linear([A;B]))，offset = Linear([A;B])。
    """
    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, dim),
            torch.nn.Sigmoid()
        )
        self.offset = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, dim),
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A, B: [B, 512] 或 [1, 512]
        cat = torch.cat([A, B], dim=-1)   # [B, 1024]
        g = self.gate(cat)                # [B, 512]
        off = self.offset(cat)            # [B, 512]
        return g * A + (1.0 - g) * B + off
