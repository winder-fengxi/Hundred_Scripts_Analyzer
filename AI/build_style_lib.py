# build_style_lib.py

import os
import io
import yaml
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data_loader.dataset import CalligraphyDataset
from author_encoder import MultiExpertEncoder, StyleLibrary

class AuthorOnlyDataset(Dataset):
    """
    只加载某位作者所有图片，用于批量提取风格向量
    """
    def __init__(self, img_paths, transform):
        self.paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 以 RGB 读取，保证 3 通道输入
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

if __name__ == "__main__":
    # 1. 加载配置（UTF-8 编码）
    with io.open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 提取配置字段
    root_dir      = cfg['data']['root_dir']
    author_subdir = cfg['data']['author_dir']
    style_lib_dir = cfg['paths']['style_lib_dir']
    style_dim     = cfg['model']['style_dim']
    batch_size    = cfg['train']['batch_size']
    img_size      = cfg['data']['img_size']

    # 3. 发现作者列表
    author_root = os.path.join(root_dir, author_subdir)
    author_ids = [
        d for d in os.listdir(author_root)
        if os.path.isdir(os.path.join(author_root, d))
    ]
    print(f"检测到 {len(author_ids)} 位作者：{author_ids}")

    # 4. 初始化风格库（若目录不存在则创建）
    style_lib = StyleLibrary(style_lib_dir)

    # 5. 初始化多专家风格编码器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    style_enc = MultiExpertEncoder(font_num=len(author_ids),
                                   style_dim=style_dim).to(device)
    style_enc.eval()

    # 6. 定义图像预处理：3 通道 + Resize + ToTensor
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),                  # 生成 [3, H, W] 的 Tensor
        transforms.Normalize(                   # 预训练 EfficientNet 通常要求归一化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 7. 遍历每位作者，用全量样本计算平均风格向量
    for author_id in author_ids:
        author_dir = os.path.join(author_root, author_id)
        img_paths = [
            os.path.join(author_dir, fn)
            for fn in sorted(os.listdir(author_dir))
            if fn.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        if not img_paths:
            print(f"[警告] 作者 {author_id} 文件夹为空，跳过")
            continue

        ds_loader = DataLoader(
            AuthorOnlyDataset(img_paths, transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        sum_vec = torch.zeros(style_dim, device=device)
        count = 0
        with torch.no_grad():
            for batch in ds_loader:
                batch = batch.to(device)           # [B,3,H,W]
                vecs = style_enc(batch)            # [B, style_dim]
                sum_vec += vecs.sum(dim=0)
                count += vecs.size(0)

        style_vec = (sum_vec / count).unsqueeze(0)  # [1, style_dim]
        style_lib.save(author_id, style_vec.cpu())
        print(f"作者[{author_id}]：{count} 张样本 平均向量 shape={style_vec.shape}")

    print("所有作者风格向量已生成并保存至：", style_lib_dir)
