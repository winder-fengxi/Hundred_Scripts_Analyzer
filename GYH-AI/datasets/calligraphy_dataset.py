# calligraphy_dataset.py
# -*- coding: utf-8 -*-
import os
import random
from PIL import Image
from torch.utils.data import Dataset

class CalligraphyDataset(Dataset):
    """
    自定义书法数据集：直接按书法家分类，train/test分割随机抽取测试样本。
    测试集大小为 num_authors * test_samples_per_author
    """
    def __init__(self, root_dir, transform=None, train=True, test_samples_per_author=500):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.test_samples_per_author = test_samples_per_author
        self.samples = []  # [(path,label_idx)]
        self.calligrapher_names = []
        self.class_to_idx = {}

        # 遍历style-author文件夹
        authors = {}
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                _, author = folder.split('-')
            except ValueError:
                continue
            # 收集所有图片
            imgs = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if imgs:
                authors.setdefault(author, []).extend(imgs)

        # 构建类名和索引
        self.calligrapher_names = sorted(authors.keys())
        self.class_to_idx = {a: i for i, a in enumerate(self.calligrapher_names)}

        # 按train/test划分样本
        for author, paths in authors.items():
            random.shuffle(paths)
            if train:
                selected = paths[self.test_samples_per_author:]
            else:
                selected = paths[:self.test_samples_per_author]
            label = self.class_to_idx[author]
            for p in selected:
                self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
