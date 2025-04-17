import torch
import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize
from skimage.util import img_as_ubyte

def skeletonize(img_tensor):
    """
    对输入的图像张量做骨架化。
    输入：
      img_tensor: torch.Tensor，形状可为 [H,W]、[1,H,W] 或 [B,1,H,W]，
                  值范围假定在 [0,1]，越大越“实”。
    返回：
      骨架化后的二值张量，跟输入同样的形状。
    """
    # 将 tensor 转成 numpy 二维数组
    need_batch = False
    x = img_tensor
    if x.dim() == 4:  # [B,1,H,W]
        need_batch = True
        batch = []
        for i in range(x.size(0)):
            skeleton = skeletonize(x[i, 0])
            batch.append(skeleton)
        return torch.stack(batch, dim=0).unsqueeze(1)

    if x.dim() == 3:  # [1,H,W]
        x = x[0]

    arr = x.detach().cpu().numpy()
    # 二值化阈值（可根据实际图像调）
    bw = arr > arr.mean()
    # skimage 的 skeletonize 要求输入是 bool 二值图
    skel = sk_skeletonize(bw)
    # 转回 [0,1] 的 float tensor
    skel = skel.astype(np.float32)
    out = torch.from_numpy(skel)
    if img_tensor.dim() == 3:
        return out.unsqueeze(0)  # [1,H,W]
    else:
        return out  # [H,W]
