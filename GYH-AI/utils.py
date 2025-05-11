# -*- coding: utf-8 -*-
import torch

def save_checkpoint(state, filename='checkpoint.pth'):
    """
    保存训练过程的检查点，包括模型状态和优化器状态等。
    state: 字典，包含 'epoch', 'model_state', 'optimizer_state'等信息
    """
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename):
    """
    从文件加载检查点，恢复模型和优化器状态。
    返回值为下一个epoch开始的索引（即加载的epoch+1）。
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded: {filename}, resume from epoch {start_epoch}")
    return start_epoch
