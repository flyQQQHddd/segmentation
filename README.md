# Semantic Segmentation

## 项目简介

适用于无人机影像的语义分割、迁移学习PyTorch框架

## 快速开始

本项目可以在[Featurize](https://featurize.cn/)平台上直接运行

```bash
# 下载数据集
featurize dataset download 4e90058c-8194-4325-83a7-69eb5c79a622
featurize dataset download ae7551ec-ac91-4cce-b039-01a04a09a19e

# 准备训练数据
7z x ./data/WHUS.zip -r
mv ./WHUS ./data
cd ./work/code
bash ./prepare.sh
cp ~/data/WHUSDataset/*csv ./dataset

# 安装 Python 依赖
pip install pytorch-ignite nvidia-ml-py albumentations
pip install nvidia-ml-py albumentations

# 使用 tmux 在后台运行程序
sudo apt install tmux
tmux new -s train
bash ./train.sh

# 分离进程
ctrl + b d
```

## 功能

### 遥感/无人机影像切割

- 基于OpenCV和PIL的多进程影像切割

### 语义分割模型训练

- DeepLab V3 Plus
- U-Net
- PSPNet
- SegNet

### 模型无监督域自适应

- AdaptSegNet：简单的基于对抗的域自适应，具有多层对抗机制
- MCDDA：考虑特定任务的决策边界来调整目标领域的分布

### 分割结果可视化与精度评定

- 对结果进行可视化，将标签图转为RGB图，同时生成缩略图，便于直接观察分割效果
- 采用mIoU和混淆矩阵进行精度评定

## 参考

[arXiv1612.01105](https://arxiv.org/abs/1612.01105)：Pyramid Scene Parsing Network

[arXiv1505.04597](https://arxiv.org/abs/1505.04597)：U-Net: Convolutional Networks for Biomedical Image Segmentation

[arXiv1505.07293](https://arxiv.org/abs/1505.07293)：SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling

[arXiv1802.02611](https://arxiv.org/abs/1802.02611)：Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

[arXiv1802.10349](https://arxiv.org/abs/1802.10349)：Learning to Adapt Structured Output Space for Semantic Segmentation

[arXiv1712.02560](https://arxiv.org/abs/1712.02560)：Maximum Classifier Discrepancy for Unsupervised Domain Adaptation

[arXiv2102.00221](https://arxiv.org/abs/2102.00221)：ObjectAug: Object-level Data Augmentation for Semantic Image Segmentation

## Tmux 使用帮助

Tmux 就是会话与窗口的 "解绑" 工具，将它们彻底分离。在云平台进行模型训练时，使用Tmux可以避免会话终止时训练也终止的尴尬局面，保证训练正常稳定进行。

```
# 创建名为 train 的会话
tmux new -s train
# 退出会话
ctrl + b d
# 查看 tmux 会话列表
tmux ls 
# 重连名为 train 的会话
tmux a -t train
# 删除名为 train 的会话
tmux kill-session -t train
```

## 更新日志

- 2024/05/24：首次提交，包含影像预处理、经典语义分割模型、AdaptSegNet和MCDDA的域自适应框架以及分割结果可视化与精度评定。
