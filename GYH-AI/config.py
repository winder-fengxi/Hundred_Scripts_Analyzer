import torch

# ==================== 超参数与路径配置 ====================
# 数据集根目录，目录下每个子文件夹为一个“书体-作者”类别，子文件夹内存放对应图片
DATA_ROOT = './chinese'

# 训练参数
BATCH_SIZE = 32         # 每个批次的样本数量
LR = 0.001              # 学习率
NUM_EPOCHS = 20         # 训练轮数
TRAIN_SPLIT = 0.8       # 训练集比例，剩余作为测试集

# 图像尺寸与归一化参数（可根据实际需要调整）
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]  # 归一化均值（ImageNet 标准）
IMG_STD  = [0.229, 0.224, 0.225]  # 归一化标准差

# 随机种子，确保可复现
SEED = 42

# 设备配置，自动选择 GPU 或 CPU
DEVICE = torch.device("cuda")

# 检查点保存路径
CHECKPOINT_DIR = './checkpoints'
# 是否从已有 checkpoint 恢复训练，如果需要请填写已有 checkpoint 路径，否则置空
RESUME = ''

# ==========================================================
