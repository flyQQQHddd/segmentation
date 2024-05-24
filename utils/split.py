'''
@ File    : split.py
@ Desc    : 切割遥感影像
@ Time    : 2024/05/08 20:50

0. PNG超大图，通过滑窗方式切割数据；
1. 滑窗同时保存窗口在原图中坐标，推理时将预测结果重新拼回PNG大图；
2. 如果内存不够大推荐用 gdal 切割数据；
'''

import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import os
from argparse import ArgumentParser
from multiprocessing import Pool
Image.MAX_IMAGE_PIXELS = 1000000000000
stride=512
target_size=(1024,1024)


if __name__ == "__main__":

    # 获取命令行参数
    parser = ArgumentParser(description="")
    parser.add_argument("-image_path",type=str)
    parser.add_argument("-label_path",type=str)
    parser.add_argument("-save_dir",type=str)
    arg = parser.parse_args()
    image_path = arg.image_path
    label_path = arg.label_path
    save_dir = arg.save_dir

    print(f"image: {image_path}")
    print(f"label: {label_path}")
    print(f"save: {save_dir}")

    # 生成输出目录
    save_image_dir = os.path.join(arg.save_dir,"image")
    save_label_dir = os.path.join(arg.save_dir,"label")
    if not os.path.isdir(save_image_dir): os.makedirs(save_image_dir)
    if not os.path.isdir(save_label_dir): os.makedirs(save_label_dir)

    # 分割文件名
    root_dir,filename = os.path.split(image_path)
    basename,filetype = os.path.splitext(filename)

    # 读取整图
    image = np.asarray(Image.open(image_path))
    if label_path is not None:
        label = np.asarray(Image.open(label_path))

    # 填充外边界至步长整数倍,方便整除
    target_w,target_h = target_size
    h,w = image.shape[0],image.shape[1]
    new_w = (w//target_w)*target_w if (w//target_w == 0) else (w//target_w+1)*target_w
    new_h = (h//target_h)*target_h if (h//target_h == 0) else (h//target_h+1)*target_h
    image = cv.copyMakeBorder(image,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,0)
    if label_path is not None:
        label = cv.copyMakeBorder(label,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,0)

    # 填充1/2 stride长度的外边框
    h,w = image.shape[0],image.shape[1]
    new_w,new_h = w + stride,h + stride
    image = cv.copyMakeBorder(image,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,0)
    if label_path is not None:
        label = cv.copyMakeBorder(label,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,0)

    cnt = 0 # 窗口计数
    csv_pos_list = [] # 窗口坐标

    def crop(cnt,crop_image,crop_label):
        '''多进程处理函数
        '''
        image_name = os.path.join(save_image_dir,basename+"_"+str(cnt)+".png")
        cv.imwrite(image_name,crop_image)
        if crop_label is not None:
            label_name = os.path.join(save_label_dir,basename+"_"+str(cnt)+".png")
            cv.imwrite(label_name,crop_label)
        
    h,w = image.shape[0],image.shape[1]
    P = Pool()
    for i in tqdm(range(w//stride-1)):
        for j in range(h//stride-1):
            topleft_x = i*stride
            topleft_y = j*stride
            crop_image = image[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
            crop_label = label[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w] if label_path is not None else None

            if crop_image.shape[:2]!=(target_h,target_h):
                print(topleft_x,topleft_y,crop_image.shape)

            # 去掉原始图像中全部空的切片
            if np.sum(crop_image) == 0:
                pass
            else:
                P.apply_async(crop,(cnt,crop_image,crop_label))
                csv_pos_list.append([basename+"_"+str(cnt)+".png",topleft_x,topleft_y,topleft_x+target_w,topleft_y+target_h])
                cnt += 1
    
    # 保存切片图像索引
    csv_pos_list = pd.DataFrame(csv_pos_list)
    csv_pos_list.to_csv(os.path.join(arg.save_dir,basename+".csv"),header=None,index=None)

    # 等待多进程执行结束
    P.close()
    P.join()