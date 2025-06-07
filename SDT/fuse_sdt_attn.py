#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python fuse_sdt_attn.py --cfg configs/CHINESE_USER.yml --pretrained_model checkpoint/checkpoint-iter199999.pth --style_dir_A ./author_chu --style_dir_B ./user --fusion_type attention --alpha 0.6 --save_dir Generated/Fused_Attn
import argparse
import os
import torch
import tqdm
from PIL import Image
from torchvision import transforms
from einops import rearrange
from glob import glob

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from models.gmm import get_seq_from_gmm
from models.model import SDT_Generator, generate_square_subsequent_mask
from utils.util import coords_render

# 融合方法统一放到 fusion_methods.py 文件里
from fusion_methods import slerp, geo_interp, attention_fuse, FusionMLP

from data_loader.loader import UserDataset


def load_style_images(style_dir: str, num_samples: int, device: torch.device):
    exts = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(style_dir, ext)))
    paths = sorted(paths)
    if len(paths) < num_samples:
        raise ValueError(f"[Error] {style_dir} 下只有 {len(paths)} 张图片，少于 {num_samples}。")
    paths = paths[:num_samples]

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    tensors = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        t = preprocess(img)
        tensors.append(t)
    style_batch = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # [1, N, 1, 256, 256]
    return style_batch


def extract_writer_style(model: SDT_Generator, style_imgs: torch.Tensor):
    model.eval()
    with torch.no_grad():
        B, N, C, H, W = style_imgs.shape
        flat = style_imgs.view(B * N, C, H, W)                  # [N,1,256,256]
        feat = model.Feat_Encoder(flat)                         # [N,512,2,2]
        feat = feat.view(B * N, 512, -1).permute(2, 0, 1)        # [4, N, 512]
        feat = model.add_position(feat)                         # [4, N, 512]
        memory = model.base_encoder(feat)                        # [4, N, 512]
        writer_memory = model.writer_head(memory)                # [4, N, 512]
        wm = rearrange(writer_memory, 't b c -> (t b) c')        # [4*N, 512]
        writer_vec = wm.mean(dim=0, keepdim=True)                # [1, 512]
        return writer_vec


def extract_glyph_style(model: SDT_Generator, style_imgs: torch.Tensor):
    model.eval()
    with torch.no_grad():
        B, N, C, H, W = style_imgs.shape
        flat = style_imgs.view(B * N, C, H, W)                  # [N,1,256,256]
        feat = model.Feat_Encoder(flat)                         # [N,512,2,2]
        feat = feat.view(B * N, 512, -1).permute(2, 0, 1)        # [4, N, 512]
        feat = model.add_position(feat)                         # [4, N, 512]
        memory = model.base_encoder(feat)                        # [4, N, 512]
        glyph_memory = model.glyph_head(memory)                  # [4, N, 512]
        gm = rearrange(glyph_memory, 't b c -> (t b) c')         # [4*N, 512]
        glyph_vec = gm.mean(dim=0, keepdim=True)                 # [1, 512]
        return glyph_vec


def collapse_to_memory(style_vec: torch.Tensor, B: int) -> torch.Tensor:
    t = 4
    if style_vec.shape[0] == 1 and B > 1:
        style_vec = style_vec.expand(B, 512)  # [B,512]
    memory = style_vec.unsqueeze(0).expand(t, B, 512).clone()  # [4, B, 512]
    return memory


def inference_with_fused_memory(model: SDT_Generator,
                                memory_writer: torch.Tensor,
                                memory_glyph: torch.Tensor,
                                char_img: torch.Tensor,
                                max_len: int = 120):
    model.eval()
    with torch.no_grad():
        B = char_img.size(0)
        char_feat = model.content_encoder(char_img)  # [4, B, 512]
        char_emb = char_feat.mean(dim=0)             # [B, 512]

        src_tensor = torch.zeros(max_len + 1, B, 512, device=char_emb.device)  # [T+1, B, 512]
        pred_sequence = torch.zeros(max_len, B, 5, device=char_emb.device)     # [T, B, 5]
        src_tensor[0] = char_emb
        tgt_mask = generate_square_subsequent_mask(max_len + 1).to(char_emb.device)

        for i in range(max_len):
            src_tensor[i] = model.add_position(src_tensor[i], step=i)
            wri_hs = model.wri_decoder(src_tensor, memory_writer, tgt_mask=tgt_mask)
            hs     = model.gly_decoder(wri_hs[-1], memory_glyph,   tgt_mask=tgt_mask)

            h_i = hs[-1][i]                   # [B, 512]
            gmm_pred = model.EmbtoSeq(h_i)    # [B, 123]
            next_step = get_seq_from_gmm(gmm_pred)  # [B, 5]
            pred_sequence[i] = next_step

            seq_emb = model.SeqtoEmb(next_step)     # [B, 512]
            src_tensor[i + 1] = seq_emb

            pen_state = next_step[:, 2:]  # [B, 3]
            if (pen_state[:, -1].sum().item() == B):
                break

        return pred_sequence.transpose(0, 1)  # [B, T, 5]


def main(opt):
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
        raise IOError("[Main] 请输入正确的 checkpoint 权重路径")

    model.eval()

    # 1. 一次性提取作者 A/B 风格向量
    style_A = load_style_images(opt.style_dir_A, opt.num_style_imgs, device)
    style_B = load_style_images(opt.style_dir_B, opt.num_style_imgs, device)
    s_writer_A = extract_writer_style(model, style_A)  # [1,512]
    s_writer_B = extract_writer_style(model, style_B)
    s_glyph_A  = extract_glyph_style(model, style_A)   # [1,512]
    s_glyph_B  = extract_glyph_style(model, style_B)

    α = opt.alpha
    if not (0.0 <= α <= 1.0):
        raise ValueError("[Main] alpha 必须在 [0,1] 之间")

    # 2. 不同 fusion_type 下的融合方式
    fusion_type = opt.fusion_type.lower()
    if fusion_type == "linear":
        s_writer_F = α * s_writer_A + (1.0 - α) * s_writer_B
        s_glyph_F  = α * s_glyph_A  + (1.0 - α) * s_glyph_B

    elif fusion_type == "slerp":
        s_writer_F = slerp(α, s_writer_A, s_writer_B)
        s_glyph_F  = slerp(α, s_glyph_A, s_glyph_B)

    elif fusion_type == "geo":
        s_writer_F = geo_interp(α, s_writer_A, s_writer_B)
        s_glyph_F  = geo_interp(α, s_glyph_A, s_glyph_B)

    elif fusion_type == "attention":
        # 注意力融合要在循环中动态计算
        s_writer_F = None
        s_glyph_F  = None

    elif fusion_type == "mlp":
        # FusionMLP 需要先自己预训练好权重，如有 checkpoint 可加载
        fusion_net = FusionMLP(dim=512, hidden=256).to(device)
        # fusion_net.load_state_dict(torch.load("fusion_mlp.pth"))
        s_writer_F = fusion_net(s_writer_A, s_writer_B)
        s_glyph_F  = fusion_net(s_glyph_A,  s_glyph_B)

    else:
        raise ValueError(f"[Main] 不支持 fusion_type={fusion_type}")

    # 3. 构建融合后的 memory（batch_size = cfg.TRAIN.IMS_PER_BATCH）
    B_max = cfg.TRAIN.IMS_PER_BATCH
    if fusion_type == "attention":
        mem_w_A = collapse_to_memory(s_writer_A, B_max)
        mem_w_B = collapse_to_memory(s_writer_B, B_max)
        mem_g_A = collapse_to_memory(s_glyph_A,  B_max)
        mem_g_B = collapse_to_memory(s_glyph_B,  B_max)
    else:
        mem_writer_F_full = collapse_to_memory(s_writer_F, B_max)
        mem_glyph_F_full  = collapse_to_memory(s_glyph_F,  B_max)

    # 4. 加载数据集（这里用 UserDataset），并设置生成上限：100 张
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

    generated_count = 0  # 全局计数器，限制最多生成 100 张

    for data in tqdm.tqdm(test_loader, desc="Generating"):
        if generated_count >= 200:
            break  # 达到 100 张时提前退出

        img_list, char_img, char_list = data["img_list"], data["char_img"].cuda(), data["char"]
        B_actual = char_img.size(0)

        # 4.1 如果用 attention 融合，则动态做 attention_fuse
        if fusion_type == "attention":
            with torch.no_grad():
                char_feat = model.content_encoder(char_img)
                char_emb  = char_feat.mean(dim=0)  # [B,512]

            mem_writer_A = mem_w_A[:, :B_actual, :]
            mem_writer_B = mem_w_B[:, :B_actual, :]
            mem_glyph_A  = mem_g_A[:,  :B_actual, :]
            mem_glyph_B  = mem_g_B[:,  :B_actual, :]

            mem_writer_F = attention_fuse(char_emb, mem_writer_A, mem_writer_B)
            mem_glyph_F  = attention_fuse(char_emb, mem_glyph_A,  mem_glyph_B)

        else:
            mem_writer_F = mem_writer_F_full[:, :B_actual, :]
            mem_glyph_F  = mem_glyph_F_full[:,  :B_actual, :]

        # 4.2 解码并保存
        preds = inference_with_fused_memory(
            model,
            mem_writer_F,
            mem_glyph_F,
            char_img,
            max_len=opt.max_len
        )  # [B_actual, T, 5]
        preds = preds.detach().cpu().numpy()

        for i in range(B_actual):
            if generated_count >= 100:
                break

            sk_pil = coords_render(
                preds[i],
                split=True, width=256, height=256,
                thickness=8, board=1
            )
            save_path = os.path.join(opt.save_dir, f"{char_list[i]}.png")
            sk_pil.save(save_path)
            generated_count += 1

    print(f"[Main] 完成，已生成 {generated_count} 张融合结果（上限 100 张）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", dest="cfg_file", default="configs/CHINESE_USER.yml",
        help="配置文件"
    )
    parser.add_argument(
        "--save_dir", dest="save_dir", default="Generated/Fused_Limited100",
        help="输出目录"
    )
    parser.add_argument(
        "--pretrained_model", dest="pretrained_model", required=True,
        help="预训练模型权重路径"
    )
    parser.add_argument(
        "--style_dir_A", dest="style_dir_A", required=True,
        help="作者 A 静态风格图目录"
    )
    parser.add_argument(
        "--style_dir_B", dest="style_dir_B", required=True,
        help="作者 B 静态风格图目录"
    )
    parser.add_argument(
        "--num_style_imgs", dest="num_style_imgs", type=int, default=20,
        help="用于提取风格的图像数量"
    )
    parser.add_argument(
        "--alpha", dest="alpha", type=float, default=0.5,
        help="融合系数 α ∈ [0,1]"
    )
    parser.add_argument(
        "--fusion_type", dest="fusion_type", default="linear",
        choices=["linear", "slerp", "geo", "attention", "mlp"],
        help="融合方式"
    )
    parser.add_argument(
        "--max_len", dest="max_len", type=int, default=120,
        help="最大解码步数"
    )
    opt = parser.parse_args()
    main(opt)
