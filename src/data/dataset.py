'''
@ File    : dataset.py
@ Desc    : 自定义 Pytorch 数据集
@ Time    : 2024/05/08 19:42
@ Info    : Map 型数据集，必须重写__len__()和__getitem__()方法
'''

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import cv2 as cv
import os

class PNGDataset(Dataset):

    def __init__(self, csv_file:str, image_dir:str, mask_dir:str=None, transforms=None) -> Dataset:
        """
        :param csv_file   : CSV 文件路径
        :param image_dir  : 图像文件所在文件夹
        :param mask_dir   : 掩膜文件所在文件夹（optional）
        :param transforms : 数据增强（optional）
        """
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):

        # 从 CSV 文件中获取文件名
        filename = self.csv_file.iloc[idx, 0]
        _, filename = os.path.split(filename)

        # 读取图像
        image_path = os.path.join(self.image_dir,filename)
        image = np.asarray(Image.open(image_path))   # mode:RGBA
        image = cv.cvtColor(image,cv.COLOR_RGBA2RGB) # PIL(RGBA)-->cv2(RGB)

        # 读取掩膜
        if self.mask_dir is not None:

            mask_path = os.path.join(self.mask_dir,filename)
            mask = np.asarray(Image.open(mask_path))     # mode:P(单通道)

        # 构建样本
        if self.mask_dir is not None:
            sample = {'image':image, 'mask':mask}
        else:
            sample = {'image':image}

        # 数据增强
        if self.transforms is not None:

            sample = self.transforms(**sample)

        return sample


class InferenceDataset(Dataset):

    def __init__(self, csv_file:str, image_dir:str, transforms=None) -> Dataset:
        """
        :param csv_file   : CSV 文件路径
        :param image_dir  : 图像文件所在文件夹
        :param transforms : 数据增强（optional）
        """

        self.image_dir = image_dir
        self.csv_file = pd.read_csv(csv_file,header=None)
        self.transforms = transforms

    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self,idx):

        filename = self.csv_file.iloc[idx,0]
        _, filename = os.path.split(filename)
        image_path = os.path.join(self.image_dir,filename)
        image = np.asarray(Image.open(image_path))   # mode:RGBA
        image = cv.cvtColor(image,cv.COLOR_RGBA2RGB) # PIL(RGBA)-->cv2(RGB)
        
        sample = {'image': image}

        if self.transforms:

            sample = self.transforms(**sample)

        image = sample['image']
        
        # ---> (topleft_x,topleft_y,buttomright_x,buttomright_y)
        pos_list = self.csv_file.iloc[idx,1:].values.astype("int")  
        return image, pos_list




