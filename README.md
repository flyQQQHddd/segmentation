

## 快速开始

```
# 下载数据集
featurize dataset download 4e90058c-8194-4325-83a7-69eb5c79a622

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

ctrl + b d
```

## Transfer 数据集

featurize dataset download ae7551ec-ac91-4cce-b039-01a04a09a19e


## tmux 使用帮助

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

## 图像尺寸
part                         26155, 25468
cropland                     6063,  4587
8cm_sun1                     41400, 27600
8cm_sun2                     34300, 22498
8cm_cloud2                   34300, 22498
8cm_after2                   34294, 22496
