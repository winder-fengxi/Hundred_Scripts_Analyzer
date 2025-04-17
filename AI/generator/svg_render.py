import io
import cv2
from matplotlib import transforms
import numpy as np
import svgwrite
from cairosvg import svg2png
from PIL import Image
import torch
import torch.nn as nn

class VectorRenderer(nn.Module):
    """
    将特征图转换为 SVG 矢量图，并渲染为位图
    步骤：
      1. 合并多通道为灰度图
      2. 二值化得到掩码
      3. 寻找轮廓（cv2.findContours）
      4. 使用 svgwrite 绘制路径
      5. cairosvg 渲染 SVG 为 PNG，再读入为 PIL
      6. 转为 Tensor
    """
    def __init__(self, output_size=(256,256), threshold=127, stroke_width=2, stroke_color='#000000'):
        super().__init__()
        self.output_size = output_size
        self.threshold = threshold
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color

    def forward(self, feat_map):
        # feat_map: Tensor [B, C, H, W]
        outputs = []
        batch = feat_map.size(0)
        # 逐样本处理
        for i in range(batch):
            x = feat_map[i].detach().cpu().numpy()  # [C,H,W]
            # 灰度合并
            img = np.mean(x, axis=0)  # [H,W]
            # 归一到0-255
            img = ((img - img.min())/(img.max()-img.min())*255).astype(np.uint8)
            # 二值化
            _, mask = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
            # 反转颜色：黑字白底 -> 白字黑底，方便轮廓提取
            mask_inv = 255 - mask
            # 提取轮廓
            contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 创建 SVG 画布
            dwg = svgwrite.Drawing(size=self.output_size)
            h, w = img.shape
            # 缩放比例
            sx = self.output_size[0] / w
            sy = self.output_size[1] / h
            for cnt in contours:
                # 简化轮廓点
                pts = cnt.squeeze().tolist()
                # 构造 path 字符串
                if len(pts) < 2:
                    continue
                path_data = f"M {pts[0][0]*sx},{pts[0][1]*sy}"
                for (px, py) in pts[1:]:
                    path_data += f" L {px*sx},{py*sy}"
                # 添加路径
                dwg.add(dwg.path(d=path_data, stroke=self.stroke_color,
                                 fill='none', stroke_width=self.stroke_width))
            # 渲染为 PNG
            svg_bytes = dwg.tostring().encode('utf-8')
            png_bytes = svg2png(bytestring=svg_bytes,
                                 output_width=self.output_size[0],
                                 output_height=self.output_size[1])
            # 从 PNG bytes 读取为 PIL Image
            pil_img = Image.open(io.BytesIO(png_bytes)).convert('L')
            # 转为 Tensor [1,H,W]
            tensor_img = transforms.ToTensor()(pil_img)
            outputs.append(tensor_img)
        # 拼 batch
        return torch.stack(outputs, dim=0)
