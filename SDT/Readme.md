# 风格融合  

本项目基于[SDT](https://github.com/dailenson/SDT)  

## 环境需求  

```python
conda update conda
conda create -n sdt python=3.8
conda activate sdt
python -m pip install --upgrade pip
# install all dependencies
#下面是下载torch库，
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install certifi
pip install charset-normalizer
pip install colorama
pip install easydict
pip install einops
pip install fastdtw
pip install filelock
pip install fsspec
pip install idna
pip install Jinja2
pip install lmdb
pip install MarkupSafe
pip install mpmath
pip install networkx
pip install numpy
pip install opencv-contrib-python
pip install packaging
pip install pillow
pip install protobuf
pip install PyYAML
pip install requests
pip install setuptools
pip install six
pip install sympy
pip install tensorboardX
pip install tqdm
pip install typing_extensions
pip install urllib3

```  

## 数据集  

中国书法作者的数据集见链接：[百度云数据集链接]( https://pan.baidu.com/s/1OOH-Xz76Aq3L9IWGddwxDQ?pwd=v4sa) 提取码: v4sa  
SDT的个人数据集构建参考：[数据集相关](https://github.com/dailenson/SDT/issues/78)。注意，本数据集对笔画宽度要求较高，建议使用细笔画。 
权重文件下载 [pth文件](https://www.123pan.com/s/TmBBjv-xDlaH.html)

## 用法  

单个作者字体生成:

```python
#为了快速生成，脚本默认只生成100个字符
python user_generate.py --pretrained_model checkpoint\checkpoint-iter199999.pth --style_path hecheng  --dir Generated/Filtered_Chars_hecheng
```  

两个作者融合生成：

```python
python fuse_sdt_attn.py --cfg configs/CHINESE_USER.yml --pretrained_model checkpoint/checkpoint-iter199999.pth --style_dir_A ./author_chu --style_dir_B ./user --fusion_type attention --alpha 0.6 --save_dir Generated/Fused_Attn
```  

其中“--fusion_type”为选择融合方法，融合方法都在fusion_methods.py中，目前支持linear，slerp，geo，attention，mlp。另，这个为限制生成个数，默认全部生成  
