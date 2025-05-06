
# train.py
# -*- coding: utf-8 -*-
import os
import random
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from datasets.calligraphy_dataset import CalligraphyDataset
from models.effnet_simplified import EfficientNetSmall
from utils import save_checkpoint

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

    # 数据集
    train_set = CalligraphyDataset(args.data_dir, transform=train_transform, train=True,
                                   test_samples_per_author=args.test_samples)
    test_set  = CalligraphyDataset(args.data_dir, transform=test_transform, train=False,
                                   test_samples_per_author=args.test_samples)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 保存映射
    mapping = {'calligrapher_names': train_set.calligrapher_names}
    with open(os.path.join(args.save_dir, 'class_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 模型
    model = EfficientNetSmall(num_classes=len(train_set.calligrapher_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        # 训练
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # 测试
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Test"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().tolist()
                all_pred.extend(preds)
                all_true.extend(labels.tolist())
        acc = accuracy_score(all_true, all_pred)
        print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")
        # 保存checkpoint
        ckpt = {'epoch':epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
        save_checkpoint(ckpt, filename=os.path.join(args.save_dir, f'epoch_{epoch}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='书法家风格识别训练')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--test_samples', type=int, default=500, help='每位作者测试样本数')
    args = parser.parse_args()
    # 保持复现
    random.seed(42)
    torch.manual_seed(42)
    main(args)
