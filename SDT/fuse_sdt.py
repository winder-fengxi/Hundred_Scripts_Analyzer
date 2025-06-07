#python fuse_sdt.py --cfg configs/CHINESE_USER.yml --save_dir Generated/Fused_Result --pretrained_model checkpoint/checkpoint-iter199999.pth --style_dir_A author_chu --style_dir_B user --alpha 0.5 --max_len 120 --num_style_imgs 20 

import argparse
import os
import torch
import tqdm
import numpy as np
from glob import glob
from PIL import Image
from torchvision import transforms
from einops import rearrange

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg

# 导入 MDN 解码需要的函数
from models.gmm import get_seq_from_gmm
# 导入 SDT_Generator 及生成掩码的函数
from models.model import SDT_Generator, generate_square_subsequent_mask
from utils.util import coords_render

# 这里直接用原项目的 UserDataset（加载灰度图/在线轨迹）
from data_loader.loader import UserDataset
import torch
import math




# ================================================
# 1. 从 style_dir 里一次性读取前 num_samples 张静态风格图 → [1, N, 1, 256, 256]
# ================================================
def load_style_images(style_dir: str, num_samples: int, device: torch.device):
    """
    从 style_dir 中读取前 num_samples 张图，转换为 [1, N, 1, 256, 256]。
    只保留 png/jpg/jpeg 文件，按文件名排序后取前 num_samples 张。
    """
    exts = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(style_dir, ext)))
    paths = sorted(paths)
    if len(paths) < num_samples:
        raise ValueError(f"[load_style_images] 文件夹 {style_dir} 中仅能找到 {len(paths)} 张图片，少于要求 {num_samples} 张。")
    paths = paths[:num_samples]

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # [1,256,256], float32 ∈ [0,1]
    ])

    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = preprocess(img)  # [1,256,256]
        tensors.append(t)
    # 合并成 [N,1,256,256]，然后 unsqueeze 成 [1,N,1,256,256]
    style_batch = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    return style_batch  # torch.Tensor, float32


# ================================================
# 2. 提取作者级别（writer-level）风格向量 → [1,512]
# ================================================
def extract_writer_style(model: SDT_Generator, style_imgs: torch.Tensor):
    """
    用 SDT_Generator 的 Feat_Encoder → base_encoder → writer_head 提取风格向量。
    输入：style_imgs = [1, N, 1, 256, 256]
    步骤：
      1.把 [1,N,1,256,256] → 展平成 [N,1,256,256]
      2.FeaT_Encoder → [N,512,2,2]
      3.reshape + position → [4, N, 512]
      4.base_encoder → [4, N, 512]
      5.writer_head → [4, N, 512]
      6.rearrange→[4*N,512]，mean → [1,512]
    返回：writer_vec [1,512]
    """
    model.eval()
    with torch.no_grad():
        B, N, C, H, W = style_imgs.shape  # B=1, N=num_samples
        flat = style_imgs.view(B * N, C, H, W)            # [N,1,256,256]
        feat = model.Feat_Encoder(flat)                   # [N,512,2,2]
        feat = feat.view(B * N, 512, -1).permute(2, 0, 1)  # [4, N, 512]
        feat = model.add_position(feat)                   # [4, N, 512]
        memory = model.base_encoder(feat)                  # [4, N, 512]
        writer_memory = model.writer_head(memory)          # [4, N, 512]
        wm = rearrange(writer_memory, 't b c -> (t b) c')  # [4*N, 512]
        writer_vec = wm.mean(dim=0, keepdim=True)          # [1, 512]
        return writer_vec


# ================================================
# 3. 提取字形级别（glyph-level）风格向量 → [1,512]
# ================================================
def extract_glyph_style(model: SDT_Generator, style_imgs: torch.Tensor):
    """
    用 SDT_Generator 的 Feat_Encoder → base_encoder → glyph_head 提取字形向量。
    输入：style_imgs = [1, N, 1, 256, 256]
    返回：glyph_vec [1,512]
    """
    model.eval()
    with torch.no_grad():
        B, N, C, H, W = style_imgs.shape
        flat = style_imgs.view(B * N, C, H, W)            # [N,1,256,256]
        feat = model.Feat_Encoder(flat)                   # [N,512,2,2]
        feat = feat.view(B * N, 512, -1).permute(2, 0, 1)  # [4, N, 512]
        feat = model.add_position(feat)                   # [4, N, 512]
        memory = model.base_encoder(feat)                  # [4, N, 512]
        glyph_memory = model.glyph_head(memory)            # [4, N, 512]
        gm = rearrange(glyph_memory, 't b c -> (t b) c')   # [4*N, 512]
        glyph_vec = gm.mean(dim=0, keepdim=True)           # [1, 512]
        return glyph_vec


# ================================================
# 4. 把风格向量 [1,512] → collapse_to_memory → [4, B, 512]
# ================================================
def collapse_to_memory(style_vec: torch.Tensor, B: int):
    """
    SDT 中，风格 memory 的时序长度 t=4 固定。
    将 style_vec=[1,512] → expand → [B,512] → unsqueeze → [4, B, 512]。
    如果 style_vec=[B,512]，就直接 unsqueeze→[1,B,512]→expand。
    """
    t = 4
    if style_vec.shape[0] == 1 and B > 1:
        style_vec = style_vec.expand(B, 512)  # [B,512]
    memory = style_vec.unsqueeze(0).expand(t, B, 512).clone()  # [4, B, 512]
    return memory


# ================================================
# 5. 用融合后的 memory + char_img → 官方 Decoder 推理 → [B, T, 5]
# ================================================
def inference_with_fused_memory(model: SDT_Generator,
                                memory_writer: torch.Tensor,
                                memory_glyph: torch.Tensor,
                                char_img: torch.Tensor,
                                max_len: int = 120):
    """
    模型推理部分类似官方 inference，但把“融合后的 memory_writer/memory_glyph”
    直接塞到 Decoder 循环，跳过了 style_img→memory 的那一段。
    输入：
        memory_writer: [4, B, 512]
        memory_glyph : [4, B, 512]
        char_img     : [B, 1, 256, 256] （UserDataset 已经给的归一化灰度/骨架）
    输出：
        pred_sequence: [B, T, 5]
    流程：
      1. content_encoder(char_img) → char_feat [4,B,512] → mean(dim=0) → char_emb [B,512]
      2. 初始化 src_tensor = zeros([max_len+1, B, 512])，src_tensor[0]=char_emb
      3. 生成 tgt_mask = generate_square_subsequent_mask(max_len+1)  # [max_len+1, max_len+1]
      4. for i in range(max_len):
           · src_tensor[i] = add_position(src_tensor[i], step=i)
           · wri_hs = wri_decoder(src_tensor, memory_writer, tgt_mask)
           · hs    = gly_decoder(wri_hs[-1], memory_glyph, tgt_mask)
           · h_i   = hs[-1][i]  # [B,512]
           · gmm_pred = EmbtoSeq(h_i) → [B,123]
           · next_step = get_seq_from_gmm(gmm_pred) → [B,5]
           · pred_sequence[i] = next_step
           · seq_emb = SeqtoEmb(next_step) → [B,512]
           · src_tensor[i+1] = seq_emb
           · 如果 next_step[:,2:].all == “落笔”（最后一个 pen_state=1），break
      5. 返回 pred_sequence.transpose(0,1) → [B, T, 5]
    """
    model.eval()
    with torch.no_grad():
        B = char_img.size(0)
        # —— 1) 内容编码 → char_emb [B,512] —— 
        char_feat = model.content_encoder(char_img)  # [4, B, 512]
        char_emb = char_feat.mean(dim=0)             # [B, 512]

        # 初始化 src_tensor 与 pred_sequence
        src_tensor = torch.zeros(max_len + 1, B, 512, device=char_emb.device)  # [T+1, B, 512]
        pred_sequence = torch.zeros(max_len, B, 5, device=char_emb.device)     # [T, B, 5]

        # 把 char_emb 放到第 0 步
        src_tensor[0] = char_emb
        # 生成 Decoder mask
        tgt_mask = generate_square_subsequent_mask(max_len + 1).to(char_emb.device)

        for i in range(max_len):
            # 给 src_tensor[i] 加位置编码
            src_tensor[i] = model.add_position(src_tensor[i], step=i)

            # 用“融合后的 memory”跑一次 Decoder
            wri_hs = model.wri_decoder(src_tensor, memory_writer, tgt_mask=tgt_mask)  # [wri_layers, T+1, B, 512]
            hs     = model.gly_decoder(wri_hs[-1], memory_glyph,   tgt_mask=tgt_mask)  # [gly_layers, T+1, B, 512]

            # 取出本步隐层
            h_i = hs[-1][i]                    # [B, 512]
            gmm_pred = model.EmbtoSeq(h_i)     # [B, 123]
            next_step = get_seq_from_gmm(gmm_pred)  # [B, 5]
            pred_sequence[i] = next_step

            # 预测下一步 embedding
            seq_emb = model.SeqtoEmb(next_step)  # [B, 512]
            src_tensor[i + 1] = seq_emb

            # 判断是否“落笔”结束（pen_state[:, -1] == 1）
            pen_state = next_step[:, 2:]  # [B, 3]
            if (pen_state[:, -1].sum().item() == B):
                break

        return pred_sequence.transpose(0, 1)  # [B, T, 5]


# ================================================
# 6. 主入口：加载模型 → 提取 A/B 风格 → 插值 → for-loop 批量生成
# ================================================
def main(opt):
    # —— 6.1 加载配置 & 构建模型 —— 
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SDT_Generator(
        num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
        num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS,
        wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
        gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS
    ).to(device)

    if len(opt.pretrained_model) > 0:
        ckpt = torch.load(opt.pretrained_model, map_location=device)
        model.load_state_dict(ckpt)
        print(f"[Main] Loaded pretrained model from {opt.pretrained_model}")
    else:
        raise IOError("[Main] 请输入正确的 checkpoint 权重路径，例如 checkpoint/checkpoint-iter199999.pth")

    model.eval()

    # —— 6.2 一次性提取作者 A/B 的风格向量 —— 
    style_A = load_style_images(opt.style_dir_A, opt.num_style_imgs, device)  # [1, N, 1, 256, 256]
    style_B = load_style_images(opt.style_dir_B, opt.num_style_imgs, device)  # [1, N, 1, 256, 256]

    s_writer_A = extract_writer_style(model, style_A)  # [1,512]
    s_writer_B = extract_writer_style(model, style_B)  # [1,512]
    s_glyph_A  = extract_glyph_style(model, style_A)   # [1,512]
    s_glyph_B  = extract_glyph_style(model, style_B)   # [1,512]

    α = opt.alpha
    if not (0.0 <= α <= 1.0):
        raise ValueError("[Main] alpha 必须在 [0,1] 之间")
    s_writer_F = α * s_writer_A + (1.0 - α) * s_writer_B  # [1,512]
    s_glyph_F  = α * s_glyph_A  + (1.0 - α) * s_glyph_B   # [1,512]

    # collapse_to_memory → full memory [4, B_max, 512]
    B_max = cfg.TRAIN.IMS_PER_BATCH
    memory_writer_F_full = collapse_to_memory(s_writer_F, B_max)  # [4, B_max, 512]
    memory_glyph_F_full  = collapse_to_memory(s_glyph_F,  B_max)  # [4, B_max, 512]

    # —— 6.3 用 UserDataset 加载“灰度/在线轨迹” —— 
    test_dataset = UserDataset(
        cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, opt.style_dir_A
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS
    )

    os.makedirs(opt.save_dir, exist_ok=True)

    data_iter = iter(test_loader)
    total_batches = len(test_loader)

    with torch.no_grad():
        for _ in tqdm.tqdm(range(total_batches), desc="Generating"):
            try:
                data = next(data_iter)
            except StopIteration:
                break

            # 原 UserDataset 返回：
            #   img_list: [B, N_style, 1, 64, 64]  （我们不需要它）
            #   char_img: [B, 1, 256, 256]         （灰度图/在线轨迹 → 归一化后）
            #   char    : list of B 个字符
            _, char_img, char_list = (
                data["img_list"], data["char_img"].cuda(), data["char"]
            )

            B_actual = char_img.size(0)
            memory_writer_F = memory_writer_F_full[:, :B_actual, :]  # [4, B_actual, 512]
            memory_glyph_F  = memory_glyph_F_full[:, :B_actual, :]   # [4, B_actual, 512]

            # 用融合后的 memory + char_img 解码
            preds = inference_with_fused_memory(
                model,
                memory_writer_F,
                memory_glyph_F,
                char_img,
                max_len=opt.max_len
            )  # [B_actual, T, 5]
            preds = preds.detach().cpu().numpy()

            # 渲染并保存
            for i in range(B_actual):
                sk_pil = coords_render(
                    preds[i],
                    split=True,
                    width=256,
                    height=256,
                    thickness=8,
                    board=1
                )
                save_path = os.path.join(opt.save_dir, f"{char_list[i]}.png")
                sk_pil.save(save_path)

    print("[Main] All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", dest="cfg_file", default="configs/CHINESE_USER.yml",
        help="配置文件 (与训练时一致)"
    )
    parser.add_argument(
        "--save_dir", dest="save_dir", default="Generated/Fused_NoSkeleton",
        help="输出融合后笔画图的目录"
    )
    parser.add_argument(
        "--pretrained_model", dest="pretrained_model", required=True,
        help="预训练模型权重路径，例如 checkpoint/checkpoint-iter199999.pth"
    )
    parser.add_argument(
        "--style_dir_A", dest="style_dir_A", required=True,
        help="作者 A 的静态风格图目录，脚本会取前 num_style_imgs 张"
    )
    parser.add_argument(
        "--style_dir_B", dest="style_dir_B", required=True,
        help="作者 B 的静态风格图目录，脚本会取前 num_style_imgs 张"
    )
    parser.add_argument(
        "--num_style_imgs", dest="num_style_imgs", type=int, default=20,
        help="每位作者用于提取风格的图像数量（默认 20）"
    )
    parser.add_argument(
        "--alpha", dest="alpha", type=float, default=0.5,
        help="风格融合系数 α ∈ [0,1]（越大越偏作者 A，默认 0.5）"
    )
    parser.add_argument(
        "--max_len", dest="max_len", type=int, default=120,
        help="生成笔画最大步数 (默认 120)"
    )
    opt = parser.parse_args()
    main(opt)