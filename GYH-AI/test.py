# test.py
# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.effnet_simplified import EfficientNetSmall

def load_mapping(save_dir):
    path = os.path.join(save_dir, 'class_mapping.json')
    with open(path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    names = mapping['calligrapher_names']
    return names

def predict(image_path, model, device, transform, class_names, topk=5):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    topk_probs, topk_idx = torch.topk(probs, topk)
    results = [(class_names[i], float(topk_probs[j])) for j,i in enumerate(topk_idx)]
    return results

def main():
    parser = argparse.ArgumentParser(description='书法家风格模型测试')
    parser.add_argument('--image',      type=str, required=True, help='待测试图像路径')
    parser.add_argument('--save_dir',   type=str, default='checkpoints', help='checkpoint 所在目录')
    parser.add_argument('--checkpoint', type=str, required=True, help='要加载的 checkpoint 文件名')
    parser.add_argument('--topk',       type=int, default=5, help='输出 Top-K 结果')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 恢复书法家列表
    class_names = load_mapping(args.save_dir)
    num_classes = len(class_names)

    # 构建模型并加载权重
    model = EfficientNetSmall(num_classes=num_classes).to(device)
    ckpt = torch.load(os.path.join(args.save_dir, args.checkpoint), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    # 预处理（与训练时 test_transform 保持一致）
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

    # 预测
    results = predict(args.image, model, device, transform, class_names, topk=args.topk)
    print(f"\n📋 Top-{args.topk} Predictions for [{args.image}]:")
    for name, prob in results:
        print(f"  - {name}: {prob:.4f}")

if __name__ == '__main__':
    main()
