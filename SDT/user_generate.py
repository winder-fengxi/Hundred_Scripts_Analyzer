#python user_generate.py --pretrained_model checkpoint\checkpoint-iter199999.pth --style_path hecheng  --dir Generated/Filtered_Chars_hecheng

import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import UserDataset
import tqdm
from utils.util import coords_render
from models.model import SDT_Generator

def main(opt):
    """加载配置并初始化模型与数据加载器"""

    # 1. 读取并解析配置文件
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    # 2. 创建 UserDataset 和 DataLoader
    test_dataset = UserDataset(
        cfg.DATA_LOADER.PATH,
        cfg.DATA_LOADER.DATASET,
        opt.style_path
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS
    )

    # 3. 确保输出目录存在
    os.makedirs(opt.save_dir, exist_ok=True)

    # 4. 构建 SDT_Generator
    model = SDT_Generator(
        num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
        num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS,
        wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
        gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS
    ).to('cuda')

    # 5. 加载预训练权重
    if len(opt.pretrained_model) > 0:
        model_weight = torch.load(opt.pretrained_model)
        model.load_state_dict(model_weight)
        print(f'Loaded pretrained model from {opt.pretrained_model}')
    else:
        raise IOError('请输入正确的 checkpoint 路径')

    model.eval()

    # 6. 设置批次数量及迭代器
    batch_samples = len(test_loader)
    data_iter = iter(test_loader)

    # 7. 初始化计数器
    generated_count = 0
    max_to_generate = 100  

    # 8. 开始推理与渲染
    with torch.no_grad():
        for _ in tqdm.tqdm(range(batch_samples), desc='生成中'):
            data = next(data_iter)
            img_list, char_img, char = (
                data['img_list'].cuda(),
                data['char_img'].cuda(),
                data['char']
            )
            # 前向推理，获得坐标序列
            preds = model.inference(img_list, char_img, 120)
            bs = char_img.shape[0]
            # 在预测序列前添加一个 SOS token
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)
            preds = preds.detach().cpu().numpy()

            for i, single_pred in enumerate(preds):
                # 如果已达生成上限，则直接退出
                if generated_count >= max_to_generate:
                    print(f'已达到生成上限 {max_to_generate}，提前退出。')
                    return

                cur_char = char[i]  # 当前字符

                # 渲染成 PIL 图像
                sk_pil = coords_render(
                    single_pred,
                    split=True,
                    width=256,
                    height=256,
                    thickness=8,
                    board=1
                )

                # 保存时文件名前缀加序号，确保不重复
                save_name = f'{generated_count + 1}_{cur_char}.png'
                save_path = os.path.join(opt.save_dir, save_name)
                try:
                    sk_pil.save(save_path)
                except Exception as e:
                    print(f'保存出错：{save_path}，字符 = {cur_char}，错误信息：{e}')
                    # 如果保存失败，不增加计数，继续下一个
                    continue

                generated_count += 1

            # 如果外循环结束也没达到10张，则继续下一个 batch
        print(f'遍历完所有 batch，最终生成了 {generated_count} 张图。')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        default='configs/CHINESE_USER.yml',
        help='Config file for training (and optionally testing)'
    )
    parser.add_argument(
        '--dir',
        dest='save_dir',
        default='Generated/Chinese_User',
        help='Target directory for storing the generated characters'
    )
    parser.add_argument(
        '--pretrained_model',
        dest='pretrained_model',
        required=True,
        help='预训练模型权重路径'
    )
    parser.add_argument(
        '--style_path',
        dest='style_path',
        default='style_samples',
        help='Directory of style sample images'
    )
    opt = parser.parse_args()
    main(opt)
