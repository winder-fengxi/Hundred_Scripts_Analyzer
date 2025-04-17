import os
import yaml
import torch
from PIL import Image
from torchvision import transforms
from author_encoder import StyleLibrary
from user_encoder import ContentEncoder, StructureNet
from generator import StyleFusion, CalligraphyGenerator
from utils import get_logger

# 读取配置
cfg = yaml.safe_load(open('config/config.yaml', 'r'))
logger = get_logger(__name__)

def infer(author_id, input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载风格向量
    style_lib = StyleLibrary(cfg['paths']['style_lib_dir'])
    style_vec = style_lib.load(author_id, device=device)
    
    # 初始化模型
    content_encoder = ContentEncoder().to(device)
    structure_net   = StructureNet().to(device)
    fusion_layer    = StyleFusion(style_dim=cfg['model']['style_dim'],
                                  content_dim=cfg['model']['content_dim'],
                                  fuse_dim=cfg['model']['fuse_dim']).to(device)
    generator       = CalligraphyGenerator().to(device)
    # 加载最新检查点
    gen_ckpt = sorted(os.listdir(cfg['paths']['checkpoints']))[-1]
    generator.load_state_dict(torch.load(os.path.join(cfg['paths']['checkpoints'], gen_ckpt)))
    generator.eval()

    # 预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor()
    ])
    img = Image.open(input_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    # 特征提取
    cont_feat = content_encoder(tensor)
    struct_feat = structure_net(cont_feat.view(1, -1))
    # 融合→生成
    fused = fusion_layer(style_vec.expand(1,-1), struct_feat)
    with torch.no_grad():
        out = generator(fused)
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transforms.ToPILImage()(out.squeeze(0).cpu()).save(output_path)
    logger.info(f"Saved generated image to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--author_id', required=True)
    parser.add_argument('--input_image', required=True)
    parser.add_argument('--output_image', required=True)
    args = parser.parse_args()
    infer(args.author_id, args.input_image, args.output_image)
