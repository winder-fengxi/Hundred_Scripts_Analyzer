# data_loader/dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import skeletonize

import numpy as np
from skimage.measure import label, regionprops

class CalligraphyDataset(Dataset):
    """
    自定义 Dataset：加载作者或用户图像，支持多字图分割与预处理
    目录结构示例：
      root/
        author/
          img1.png
          img2.jpg
        user/
          upload1.png (可能含多字)
    参数:
      root: 根目录，如 "./dataset"
      mode: "author" 或 "user"
      transform: torchvision.transforms，对每张单字图做相同预处理
      split_chars: 当 mode='user' 且 split_chars=True 时，对多字图返回多张单字 Tensor
    """
    def __init__(self, root, mode='author', transform=None, split_chars=False):
        super().__init__()
        # 数据根路径 + 模式子目录
        self.root = os.path.join(root, mode)
        # 收集所有支持的图像路径
        self.paths = [
            os.path.join(self.root, fname)
            for fname in sorted(os.listdir(self.root))
            if fname.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        # 默认预处理：灰度化 + 缩放到 64×64 + ToTensor
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.split_chars = split_chars

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('L')  # 转灰度方便分割

        if self.split_chars:
            # 1. 先骨架化，得到二值骨架图 Tensor [1,H,W]
            skel_tensor = skeletonize(torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            )).squeeze(0)  # [H,W]
            skel = (skel_tensor.numpy() > 0.5).astype(np.uint8)

            # 2. 连通域标记
            lbl = label(skel, connectivity=2)
            chars = []
            for region in regionprops(lbl):
                # 过滤过小区域
                if region.area < 50:
                    continue
                # 获取包围盒
                minr, minc, maxr, maxc = region.bbox
                # 裁剪原图
                char_img = img.crop((minc, minr, maxc, maxr))  # PIL 图
                # 预处理
                char_t = self.transform(char_img)  # [1,64,64]
                chars.append(char_t)

            if len(chars) == 0:
                # 未分割出字符时，退化为整图
                chars = [ self.transform(img) ]

            # 返回 Tensor[N,1,64,64]
            return torch.stack(chars, dim=0)

        else:
            # 不分割，直接预处理整张图
            return self.transform(img)
