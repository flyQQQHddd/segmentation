import os
import torch
import torch.nn as nn
import numpy as np
import PIL

def dir_exists(path):
    if not os.path.exists(path):
            os.makedirs(path)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
            center = factor - 1
    else:
            center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def set_trainable_attr(m,b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b

def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c)>0:
        for l in c:
            apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def create_zeros_png(image_w, image_h):
    '''Description:
        0. 先创造一个空白图像，将滑窗预测结果逐步填充至空白图像中；
        1. 填充右下边界，使原图大小可以杯滑动窗口整除；
        2. 膨胀预测：预测时，对每个(1024,1024)窗口，每次只保留中心(512,512)区域预测结果，每次滑窗步长为512，使预测结果不交叠；
    ''' 

    new_h, new_w = (image_h//1024+1)*1024,(image_w//1024+1)*1024 #填充右边界
    zeros = (256+new_h+256,256+new_w+256)  #填充空白边界
    zeros = np.zeros(zeros, np.uint8)
    return zeros