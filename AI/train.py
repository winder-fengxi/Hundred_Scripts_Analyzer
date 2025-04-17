# train.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from author_encoder import MultiExpertEncoder, StyleLibrary
from user_encoder import ContentEncoder, StructureNet
from generator import StyleFusion, CalligraphyGenerator
from data_loader.dataset import CalligraphyDataset
from utils import get_logger

# 读取配置
cfg = yaml.safe_load(open('config/config.yaml', 'r'))
logger = get_logger(__name__, log_file=os.path.join(cfg['paths']['logs'], 'train.log'))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据集和 DataLoader
    train_ds = CalligraphyDataset(root=cfg['data']['root_dir'],
                                  mode=cfg['data']['author_dir'],
                                  transform=None,
                                  split_chars=False)
    loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'],
                        shuffle=True, num_workers=4)

    # 模型初始化
    author_encoder = MultiExpertEncoder(font_num=cfg['model']['num_authors'],
                                        style_dim=cfg['model']['style_dim']).to(device)
    content_encoder = ContentEncoder().to(device)
    structure_net   = StructureNet().to(device)
    fusion_layer    = StyleFusion(style_dim=cfg['model']['style_dim'],
                                  content_dim=cfg['model']['content_dim'],
                                  fuse_dim=cfg['model']['fuse_dim']).to(device)
    generator       = CalligraphyGenerator().to(device)
    discriminator   = nn.Conv2d(1,1,1).to(device)  # placeholder

    # 优化器
    params_G = list(content_encoder.parameters()) + \
               list(structure_net.parameters()) + \
               list(fusion_layer.parameters()) + \
               list(generator.parameters())
    optimizer_G = optim.Adam(params_G, lr=cfg['train']['learning_rate'],
                             betas=(cfg['train']['beta1'], cfg['train']['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg['train']['learning_rate'],
                             betas=(cfg['train']['beta1'], cfg['train']['beta2']))

    # 损失
    adversarial_loss  = nn.MSELoss()
    reconstruction_loss = nn.L1Loss()

    # 风格库
    style_lib = StyleLibrary(cfg['paths']['style_lib_dir'])

    # 训练循环
    for epoch in range(cfg['train']['num_epochs']):
        for imgs in loader:
            imgs = imgs.to(device)  # [B,1,64,64]
            # 随机选作者风格向量
            author_id = torch.randint(0, cfg['model']['num_authors'], (1,)).item()
            style_vec = style_lib.load(author_id, device=device)  # [1,style_dim]
            style_vec = style_vec.expand(imgs.size(0), -1)

            # 内容与结构特征
            cont_feat = content_encoder(imgs)
            struct_feat = structure_net(cont_feat.view(cont_feat.size(0), -1))

            # 融合→生成
            fused = fusion_layer(style_vec, struct_feat)
            gen_imgs = generator(fused)

            # 判别器/生成器训练略，使用 recon loss
            loss_G = reconstruction_loss(gen_imgs, imgs)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        logger.info(f'Epoch {epoch+1}/{cfg["train"]["num_epochs"]} Loss_G: {loss_G.item():.4f}')
        # 保存模型检查点
        torch.save(generator.state_dict(), os.path.join(cfg['paths']['checkpoints'], f'gen_epoch{epoch+1}.pth'))

if __name__ == '__main__':
    os.makedirs(cfg['paths']['logs'], exist_ok=True)
    os.makedirs(cfg['paths']['checkpoints'], exist_ok=True)
    train()
