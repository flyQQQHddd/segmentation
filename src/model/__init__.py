'''
Author      : Qu Haodong
Connect     : 2021302131044@whu.edu.cn
LastEditors : Qu Haodong
Description :  
    通过配置文件里的“model”信息，构造网络。
    网络分为三部分：
        1. Backbone：提取特征
        2. Head：分类器、回归器
        3. Loss：损失函数（迁移学习的Loss不包含在模型中）
    支持的网络：
        - deeplabv3plus
        - deeplabv3plus_pro (with transformer)
        - FCDiscriminator (a simple discriminator)
        - UNet
        - PSPNet
LastEditTime: 2024-05-08
'''

from copy import deepcopy
import torch
from . import network as Net
from torch.nn import Module
import src.model.loss as Losses


def build_model(cfg:dict, pretrain_path:str=None)->Module:
    """Build model by config

    Args:
        cfg (dict): _description_
        pretrain_path (str, optional): _description_. Defaults to None.

    Raises:
        KeyError: _description_

    Returns:
        model: torch.nn.Module
    """
    
    cfg = deepcopy(cfg)
    # 根据配置文件构造网络
    if 'net' in cfg.keys():
        net_cfg = cfg['net']
        net_type = net_cfg.pop("type")
        model = getattr(Net,net_type)(cfg)
    else:
        raise KeyError("`net` not in cfg, Got {}".format(net_type))
    
    # 读取训练好的的模型权重
    if pretrain_path:
        model_state_dict = model.state_dict()
        state_dict = torch.load(pretrain_path,map_location='cpu')

        if 'model' in state_dict.keys():
            state_dict = state_dict['model']

        for key in state_dict.keys():
            if key in model_state_dict.keys() and state_dict[key].shape==model_state_dict[key].shape:
                model_state_dict[key] = state_dict[key]

        model.load_state_dict(model_state_dict)
        print(f'Load Pretrained Model: {pretrain_path}')
    else:
        # print('No Pretrain Model')
        pass

    return model



def build_loss(cfg_loss):
    loss_type = cfg_loss.pop('type')
    if hasattr(Losses,loss_type):
        criterion = getattr(Losses,loss_type)(**cfg_loss)
        return criterion
    else:
        raise ValueError("\'type\' of loss is not defined. Got {}".format(loss_type))


