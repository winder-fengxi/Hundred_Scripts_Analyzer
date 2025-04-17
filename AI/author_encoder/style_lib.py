import os
import torch

class StyleLibrary:
    """管理和加载多位书法家预计算风格向量的工具类"""
    def __init__(self, lib_dir):
        self.lib_dir = lib_dir
        os.makedirs(lib_dir, exist_ok=True)

    def save(self, author_id, style_vector):
        """
        保存某位作者的风格向量到库中。
        author_id: 字符串或整数
        style_vector: torch.Tensor (1, style_dim)
        """
        path = os.path.join(self.lib_dir, f"{author_id}.pt")
        torch.save(style_vector.cpu(), path)

    def load(self, author_id, device=None):
        """
        从库中加载某位作者的风格向量。
        返回 torch.Tensor 在指定 device 上
        """
        path = os.path.join(self.lib_dir, f"{author_id}.pt")
        vec = torch.load(path, map_location=device)
        return vec

    def list_authors(self):
        """列出所有已保存的作者 ID（文件名不含扩展名）"""
        files = os.listdir(self.lib_dir)
        return [os.path.splitext(f)[0] for f in files]