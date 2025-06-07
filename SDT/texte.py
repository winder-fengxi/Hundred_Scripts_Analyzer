#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用法示例：
    python texte.py --input_dir ./author --output_dir ./images_binary --thresh 127 --max_iter 4
"""

import os
import glob
import argparse

import cv2
import numpy as np
from skimage.morphology import thin

def calligraphy_to_thin(
    img_gray: np.ndarray,
    thresh: int = 127,
    max_iter: int = 15
) -> np.ndarray:
    """
    将单张灰度图进行二值化，然后做有限次形态学细化，最后反转颜色（白底黑线），
    返回处理后的二值图。

    参数:
        img_gray:  输入的灰度图（numpy.ndarray，dtype=np.uint8）
        thresh:    二值化阈值（0-255），灰度 < thresh → 视为笔画（前景）
        max_iter:  形态学细化的最大迭代次数。值越大，笔画越接近骨架；值越小，保留笔画宽度越多。

    返回值:
        inverted_img: 白底黑线的细化结果（numpy.ndarray，dtype=np.uint8，取值 0 或 255）
    """

    # 1. 二值化（先反转，使笔画部分变成白 255，背景变成黑 0）
    _, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY_INV)
    binary_bool = (binary > 0)  # True 表示笔画

    # 2. 形态学细化（morphological thinning）
    #    使用 skimage.morphology.thin，可以设置 max_num_iter 让细化只做有限次迭代，保留更多笔画宽度。
    #    例如 max_iter=15 时，大多数宽度较细的笔画会保留 2-3 像素的宽度；如果把 max_iter 越调越大，会逐渐逼近骨架。
    thin_bool = thin(binary_bool, max_num_iter=max_iter)

    # 3. 转回 0/255 的 uint8 图像（此时笔画对应 255，背景对应 0）
    thin_img = (thin_bool.astype(np.uint8) * 255)

    # 4. 反转颜色：255 → 0 (黑线), 0 → 255 (白底)
    inverted_img = cv2.bitwise_not(thin_img)

    return inverted_img


def process_folder_thin(input_dir: str, output_dir: str, thresh: int, max_iter: int):
    """
    批量处理输入文件夹中的所有图片，并将“白底黑线”保留更多细节的细化结果
    保存到输出文件夹。

    参数:
        input_dir:  包含原始书法图的文件夹路径
        output_dir: 保存细化结果的文件夹路径
        thresh:     二值化阈值
        max_iter:   形态学细化的最大迭代次数
    """

    # 1. 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 2. 支持的图片后缀列表
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    # 3. 统计输入文件夹中所有图片数量
    total_files = []
    for ext in exts:
        total_files.extend(glob.glob(os.path.join(input_dir, ext)))
    total_count = len(total_files)

    if total_count == 0:
        print(f"[警告] 在 '{input_dir}' 中未找到任何支持的图片文件。")
        return

    print(f"[信息] 在 '{input_dir}' 中共找到 {total_count} 张图片，开始批量细化（保留更多细节）……")
    processed = 0

    # 4. 逐个读取、处理并保存
    for filepath in total_files:
        fname = os.path.basename(filepath)
        name, _ = os.path.splitext(fname)
        out_path = os.path.join(output_dir, f"{name}_thin_{max_iter}.png")

        # 4.1 读取为灰度图
        img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"  [跳过] 无法读取文件：{filepath}")
            continue

        # 4.2 对图像做形态学细化并反转颜色
        thin_img = calligraphy_to_thin(img_gray, thresh, max_iter)

        # 4.3 保存结果
        success = cv2.imwrite(out_path, thin_img)
        if not success:
            print(f"  [错误] 保存失败：{out_path}")
        else:
            processed += 1
            print(f"  [已处理] {fname} → {os.path.basename(out_path)}")

    print(f"[完成] 共处理 {processed}/{total_count} 张图片，结果保存在 '{output_dir}'。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量对文件夹中的书法图进行有限次形态学细化 (Thinning)，保留更多笔画细节"
    )
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        help="待处理的输入文件夹路径（包含原始书法图）"
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="保存细化结果的输出文件夹路径"
    )
    parser.add_argument(
        "--thresh", "-t",
        type=int,
        default=127,
        help="二值化阈值（0-255），默认为 127。灰度 < thresh → 视为笔画（前景）"
    )
    parser.add_argument(
        "--max_iter", "-m",
        type=int,
        default=15,
        help="形态学细化（thinning）的最大迭代次数，默认为 15。"
             "数值越大，最终笔画越接近骨架；数值越小，保留笔画宽度越多。"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("====================================================")
    print(" 批量形态学细化脚本 (Thinning, keep more details) ")
    print("====================================================")
    print(f" 输入文件夹：{args.input_dir}")
    print(f" 输出文件夹：{args.output_dir}")
    print(f" 二值化阈值：{args.thresh}")
    print(f" 细化迭代数：{args.max_iter}")
    print(" 开始处理，请稍候...\n")

    process_folder_thin(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        thresh=args.thresh,
        max_iter=args.max_iter
    )