# 中国书法风格迁移项目报告

## 一、整体架构

本系统分为三大模块，通过“离线风格提取 → 联合训练 → 在线推理”三阶段完成用户字体与书法家风格的融合：

### 系统架构概览

%%{init: {'flowchart': {'htmlLabels': true}}}%%
flowchart LR
  subgraph "离线阶段：风格向量构建"
    A["作者图像\n(dataset/author/AuthorID/*)"] --> B["多专家风格编码器"]
    B --> C["风格向量库\n(输出 .pt 文件)"]
  end

  subgraph "在线训练/微调"
    C --> D["训练循环\n(train.py)"]
    A --> D
    D --> E["保存模型检查点"]
  end

  subgraph "在线推理"
    U["用户上传图像\n(dataset/user/upload.png)"] --> V["字符分割"]
    V --> W["内容编码器"]
    C --> X["特征融合"]
    W --> X
    X --> Y["书法生成器"]
    Y --> Z["输出图像\n(outputs/result.png)"]
  end

  style A fill:#f9f,stroke:#333,stroke-width:1px
  style U fill:#9ff,stroke:#333,stroke-width:1px

**说明**：

1. **离线阶段**  
   - **MultiExpertEncoder** 对每位作者全部图像提取风格向量并平均，存入 `style_lib/AuthorID.pt`。  
2. **在线训练/微调**  
   - 在 `train.py` 中加载作者风格向量和原始图像，交替优化 Generator/Discriminator，保存检查点。  
3. **在线推理**  
   - 用户上传图片 → 连通域分割成单字 → `ContentEncoder` 提取内容特征  
   - 从 `style_lib` 加载目标作者风格向量 → `StyleFusion` 融合内容/风格  
   - `CalligraphyGenerator` 渲染并模拟墨迹扩散，输出最终书法图像  


1. **作者风格提取（author_encoder）**  
   - `MultiExpertEncoder`：基于 EfficientNet‐B3 多专家网络，对每位书法家所有 3k+ 张样本做特征编码与平均，生成一个 **固定风格向量** 并保存至 `style_lib/AuthorID.pt`。

2. **用户内容提取（user_encoder）**  
   - `ContentEncoder`：ResNet‑50 + 骨架化预处理，提取“视觉”与“结构”双分支特征。  
   - `StructureNet`：专门对二值骨架图进一步卷积与全连接，得到笔画结构向量。

3. **特征融合与渲染（generator）**  
   - `StyleFusion`：跨模态融合层，用可学习的门控 (γ) 将“作者风格向量”与“用户结构向量”智能融合。  
   - `CalligraphyGenerator`：UpBlock 上采样→ ACmix 注意力卷积→ VectorRenderer SVG 渲染→ 墨迹扩散模拟，输出最终书法风格图像。

4. **数据加载与配置**  
   - `CalligraphyDataset`：从 `dataset/author/AuthorID/` 和 `dataset/user/` 加载图像，支持多字图分割。  
   - 全局配置 (`config/config.yaml`) 统一管理路径、尺寸、超参等。

---
  
## 二、核心算法

1. **EfficientNet‐B3 多专家编码**  
   - 多路 EfficientNet 共同提取特征 → 拼接 → 交叉注意力映射 → 512 维风格向量。

2. **骨架化 + ResNet‑50 内容编码**  
   - 先用 skimage 骨架化算法提取二值笔画 → 同步喂入 ResNet 模型 + 结构映射网络，分离“形状”与“笔画”信息。

3. **门控特征融合 (StyleFusion)**  
   - 自适应门控：  
     \[
       \text{gate} = \sigma(\gamma\,W_s s + W_c c),\quad
       f = \text{gate}\odot (W_s s) + (1-\text{gate})\odot (W_c c)
     \]

4. **SVG 矢量渲染 + 墨迹物理模拟**  
   - 从融合特征生成灰度位图 → OpenCV 轮廓提取 → `svgwrite` 生成矢量 `<path>` → `cairosvg` 渲染回位图 → 蒙版式墨迹扩散模拟。

5. **训练损失（若做微调）**  
   - 重建损失：L1 内容保真  
   - 对抗损失：GAN 提升真实感（可选）  
   - 风格一致损失：风格编码器输出一致性

---

## 三、项目运行流程

### 1. 环境准备  
```bash
git clone https://.../calligraphy.git
cd calligraphy
pip install -r requirements.txt  

### 2. 数据结构  
dataset/
├── author/
│   ├── 于右任/    # 每个作者目录
│   │   ├── img0001.png
│   │   └── ... (3k+)
│   └── ... 
└── user/
    └── upload.png  # 用户多字图  
### 3. 初始化
python build_style_lib.py
# 输出 style_lib/AuthorID.pt

### 4.训练微调
python train.py --config config/config.yaml
# 载入 style_lib 向量，迭代优化 Generator/Discriminator

### 5. 推理示例
python infer.py \
  --config config/config.yaml \
  --author_id 于右任 \
  --input_image dataset/user/upload.png \
  --output_image outputs/result.png
