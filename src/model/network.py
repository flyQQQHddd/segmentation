'''
Author      : Qu Haodong
Connect     : 2021302131044@whu.edu.cn
LastEditors : Qu Haodong
Description : 
    支持的网络：
        - Deeplabv3plus
        - Deeplabv3plusTransformer (Deeplabv3plus with Transformer)
        - FCDiscriminator (a simple discriminator)
        - UNet
        - PSPNet
LastEditTime: 2024-01-30
'''

import math
import src.model.backbone as Backbones
import src.model.head as Heads

from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch
from src.utils.helpers import initialize_weights, set_trainable


def build_backbone(cfg_backbone):
    backbone_type = cfg_backbone.pop('type')
    if hasattr(Backbones,backbone_type):
        backbone = getattr(Backbones,backbone_type)(**cfg_backbone)
        return backbone
    else:
        raise ValueError("\'type\' of backbone is not defined. Got {}".format(backbone_type))

def build_head(cfg_head):
    head_type = cfg_head.pop('type')
    if hasattr(Heads,head_type):
        head = getattr(Heads,head_type)(**cfg_head)
        return head
    else:
        raise ValueError("\'type\' of head is not defined. Got {}".format(head_type))
    
    
class Deeplabv3plus(nn.Module):
    def __init__(self,cfg):
        super(Deeplabv3plus,self).__init__()

        # 获取配置
        cfg_model = deepcopy(cfg)
        cfg_backbone = cfg_model['backbone']
        cfg_head = cfg_model['head']

        # 获取类别数量
        self.num_classes = cfg_model['net']['num_classes']

        # 加载自定义的局部结构
        self.backbone = build_backbone(cfg_backbone)
        self.head = build_head(cfg_head)

        # 设置通用的局部结构
        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1),
            nn.ReLU())
        self.dropout1 = nn.Dropout(0.5)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv1x1 = nn.Sequential(nn.Conv2d(2048,256,1,bias=False),nn.ReLU())
        self.conv3x3 = nn.Sequential(nn.Conv2d(512,self.num_classes,1),nn.ReLU())   
        self.dec_conv = nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU())
       

    def forward(self, x):

        # High-Level Features 
        x = self.backbone(x) # Backbone（ResNet）
        out1 = self.head(x) # Head（ASPP）
        out1 = self.out1(out1) # 1x1 Conv
        out1 = self.dropout1(out1) # 随机丢弃层
        out1 = self.up4(out1) # Unsample by 4

        # Low-Level Features 
        dec = self.conv1x1(x) # Low-Level Features 
        dec = self.dec_conv(dec) # 低层特征卷积
        dec = self.up4(dec) # Unsample by 4

        # output
        contact = torch.cat((out1,dec),dim=1) # concat
        out = self.conv3x3(contact) # 3x3 Conv
        out = self.up4(out) # Unsample by 4

        return out

        
class FCDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(FCDiscriminator, self).__init__()

        cfg_model = deepcopy(cfg)
        self.num_classes = cfg_model['net']['num_classes']
        self.ndf = cfg_model['net']['ndf']

        self.conv1 = nn.Conv2d(self.num_classes, self.ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(self.ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


class _UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_UNetEncoder, self).__init__()

        inner_channels = out_channels // 2 if inner_channels is None else inner_channels
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)


    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x


class _UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_UNetDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)

        if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True)
            else:
                # Padding in case the incomping volumes are of different sizes
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2))

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()

        # 获取类别数量
        cfg_model = deepcopy(cfg)
        self.num_classes = cfg_model['net']['num_classes']

        self.start_conv = x2conv(3, 64)
        self.down1 = _UNetEncoder(64, 128)
        self.down2 = _UNetEncoder(128, 256)
        self.down3 = _UNetEncoder(256, 512)
        self.down4 = _UNetEncoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)

        self.up1 = _UNetDecoder(1024, 512)
        self.up2 = _UNetDecoder(512, 256)
        self.up3 = _UNetDecoder(256, 128)
        self.up4 = _UNetDecoder(128, 64)
        self.final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1)
        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x, targets=None):

        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)

        if self.training:
            loss = self.loss(x, targets)
            return loss
        else:
            return x


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, cfg):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d

        # 获取配置
        cfg_model = deepcopy(cfg)
        cfg_backbone = cfg_model['backbone']

        # 获取网络配置
        self.num_classes = cfg_model['net']['num_classes']
        self.use_aux  = cfg['net']['use_aux']
        freeze_bn  = cfg['net']['freeze_bn']
        freeze_backbone  = cfg['net']['freeze_backbone']
        self.aux_loss_weight = cfg['net']['aux_loss_weight']

        # 加载自定义的局部结构
        model = build_backbone(cfg_backbone)

        m_out_sz = 2048

        self.initial = nn.Sequential(*list(model.children())[0][:4])
        self.initial = nn.Sequential(*self.initial)

        self.layer1 = list(model.children())[0][4]
        self.layer2 = list(model.children())[0][5]
        self.layer3 = list(model.children())[0][6]
        self.layer4 = list(model.children())[0][7]

        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz//4, self.num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz//2, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz//4, self.num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)

        if freeze_bn: 
            self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)


    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def forward(self, x, targets=None):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]

        # when training
        if self.training:
            loss = self.loss(output,targets)
            if self.use_aux:
                aux = self.auxiliary_branch(x_aux)
                aux = F.interpolate(aux, size=input_size, mode='bilinear')
                aux = aux[:, :, :input_size[0], :input_size[1]]
                aux_loss = self.loss(aux, targets)
                return loss + aux_loss * self.aux_loss_weight
            else:
                return loss
        # when testing
        else:
            return output


class deeplabv3plus_mcdda(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus_mcdda,self).__init__()

        # 获取配置
        cfg_model = deepcopy(cfg)
        cfg_backbone = cfg_model['backbone']
        cfg_head = cfg_model['head']

        # 获取类别数量
        self.num_classes = cfg_model['net']['num_classes']

        # 加载自定义的局部结构
        self.backbone = build_backbone(cfg_backbone)
        self.head = build_head(cfg_head)

        # 设置通用的局部结构
        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1),
            nn.ReLU())
        self.dropout1 = nn.Dropout(0.5)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv1x1 = nn.Sequential(nn.Conv2d(2048,256,1,bias=False),nn.ReLU())
        self.dec_conv = nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU())

        self.seg=nn.Conv2d(512,self.num_classes,kernel_size=1,bias=True)
        m=self.seg
        n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
        m.weight.data.normal_(0, math.sqrt(2./n))
        m.bias.data.zero_()

    def forward(self,x):

        # High-Level Features 
        x = self.backbone(x) # Backbone（ResNet）
        out1 = self.head(x) # Head（ASPP）
        out1 = self.out1(out1) # 1x1 Conv
        out1 = self.dropout1(out1) # 
        out1 = self.up4(out1) # Unsample by 4

        # Low-Level Features 
        dec = self.conv1x1(x) # Low-Level Features 
        dec = self.dec_conv(dec) # 
        dec = self.up4(dec) # Unsample by 4

        # output
        contact = torch.cat((out1,dec),dim=1) # concat
        out = self.seg(contact) 

        return out


#梯度反转
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    def forward(self, x):
        return x.view_as(x)
    def backward(self, grad_output):
        return (grad_output*-self.lambd)
    
def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

#分类器
class DRNSegPixelClassifier(nn.Module):
    def __init__(self, cfg):
        super(DRNSegPixelClassifier, self).__init__()

        # 获取配置
        cfg_model = deepcopy(cfg)
        # 获取类别数量
        self.num_classes = cfg_model['net']['num_classes']
        self.use_torch_up = cfg_model['net']['use_torch_up']

        if self.use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            self.up = nn.ConvTranspose2d(
                self.num_classes, self.num_classes, 12, stride=4, padding=4,
                output_padding=0, groups=self.num_classes,
                bias=False)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
            x = self.up(x)
        else:
            x = self.up(x)
        return x