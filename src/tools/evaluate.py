'''
Author      : Qu Haodong
Connect     : 2021302131044@whu.edu.cn
LastEditors : Qu Haodong
Description : 评估推理结果
LastEditTime: 2024-01-30
'''
from argparse import ArgumentParser
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = 1e10


if __name__ == "__main__":

    # 获取参数
    parser = ArgumentParser(description="Pytorch Evaluate")
    parser.add_argument("--predict", '-p', type=str, help='预测结果')
    parser.add_argument("--label", '-l', type=str, help='真正标签')


    arg = vars(parser.parse_args())

    # 使用PIL读取图像
    predict:np.ndarray = np.asarray(Image.open(arg['predict']))
    label:np.ndarray = np.asarray(Image.open(arg['label']))

    # 输出图像信息
    num_class = predict.max()
    print(arg['predict'])
    print(f'Class number: {num_class}')

    # -------------------------------------------------------------------------
    # 计算IoU
    IoU_list = []
    for i in range(1, num_class + 1):
        predict_i = (predict == i)
        label_i = (label == i)
        logical_and = np.logical_and(predict_i, label_i)
        logical_or = np.logical_or(predict_i, label_i)
        logical_and_sum = np.sum(logical_and)
        logical_or_sum = np.sum(logical_or)
        IoU = logical_and_sum / logical_or_sum
        IoU_list.append(IoU)
        print(f'Class Index: {i}, IoU: {IoU}')
    print(f'MIoU: {np.mean(IoU_list)}')

    # -------------------------------------------------------------------------
    # 计算混淆矩阵
    result = []
    for i in range(0, num_class + 1):
        result.append((predict == i, label == i))

    sum = []
    for i in range(num_class+1):
        logical_and_sum_list = []
        for j in range(num_class+1):
            logical_and_sum = np.sum(np.logical_and(result[i][0], result[j][1]))
            logical_and_sum_list.append(logical_and_sum)
        print(logical_and_sum_list)
        sum.append(np.sum(logical_and_sum_list))
    print(f'Sum of pixel: {np.sum(sum)}')


