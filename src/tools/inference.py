
# 重新设置工作目录
import sys, os
sys.path[0] = os.path.join(sys.path[0], "../..")

from src.data import build_dataloader
from configs import load_arg
from src.model import build_model
import torch
from src.utils import get_free_device_ids, create_zeros_png
from datetime import datetime
import os
from torch.cuda.amp import autocast
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

Image.MAX_IMAGE_PIXELS = 1000000000000000000




def tta_forward(dataloader, model, png_shape, device):
    
    image_w,image_h = png_shape
    predict_png = create_zeros_png(image_w,image_h)
    model = model.eval()

    with torch.no_grad():

        for (image, pos_list) in tqdm(dataloader):

            image = image.float().cuda(device) 

            # # 前向传播
            # with autocast():

            #     # 进行 4 次翻转预测
            #     predict_1 = model(image)
            #     predict_2 = model(torch.flip(image,[-1]))
            #     predict_2 = torch.flip(predict_2,[-1])
            #     predict_3 = model(torch.flip(image,[-2]))
            #     predict_3 = torch.flip(predict_3,[-2])
            #     predict_4 = model(torch.flip(image,[-1,-2]))
            #     predict_4 = torch.flip(predict_4,[-1,-2])

            #     # 预测结果取平均
            #     predict = predict_1 + predict_2 + predict_3 + predict_4   

            # 前向传播
            with autocast():
                predict = model(image)
            
            predict_list = torch.argmax(predict.cpu(),1).byte().numpy()

            # 构造输出图像
            for i in range(predict_list.shape[0]):

                predict = predict_list[i]
                [topleft_x,topleft_y,buttomright_x,buttomright_y] = pos_list[i,:]
                
                if (buttomright_x-topleft_x)==1024 and (buttomright_y-topleft_y)==1024:
                    # 每次预测只保留图像中心(512,512)区域预测结果
                    predict_png[topleft_y+256:buttomright_y-256,topleft_x+256:buttomright_x-256] = predict[256:768,256:768]
                else:
                    raise ValueError("target_size!=512， Got {},{}".format(buttomright_x-topleft_x,buttomright_y-topleft_y))
    
    h,w = predict_png.shape
    predict_png =  predict_png[256:h-256,256:w-256] # 去除整体外边界
    predict_png = predict_png[:image_h,:image_w]    # 去除补全512整数倍时的右下边界
    return predict_png


if __name__ == "__main__":

    arg = vars(load_arg())

    # 获取配置文件
    cfg = None
    config_file = arg["config_file"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')
    exec(r"from {} import config as cfg".format(config_file))

    # 参数融合，优先级：arg > cfg
    for key, value in arg.items():
        cfg[key] = value

    # 创建输出目录
    now = datetime.now()
    time_str = now.strftime(r'%Y%m%d%H%M%S_')
    cfg['out_dir'] = os.path.join(cfg['out_dir'], time_str + cfg['tag'])

    # 创建目录
    if not os.path.isdir(cfg['out_dir']): os.makedirs(cfg['out_dir'])
    print("Output Directory :",cfg['out_dir'])

    # 构建推理组件
    model = build_model(cfg['model'], pretrain_path=arg['load_path']).eval()
    image_pipeline = build_dataloader(cfg['inference_pipeline'])
    
    # 设置推理显卡
    free_device_ids = get_free_device_ids()
    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids)>=max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]
    master_device = free_device_ids[0]

    # 模型结构稳定可以加快训练速度
    if cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True

    # -------------------------------------------------------------------
    # 开始推理

    # 前向传播执行推理
    model = model.cuda(master_device)

    interface_result = tta_forward(
        image_pipeline,
        model,
        cfg['inference_pipeline']['shape'],
        master_device)
    
    # 输出推理结果
    pil_image = Image.fromarray(interface_result)
    pil_image.save(os.path.join(cfg['out_dir'], "output.png"))

    # End main
    del image_pipeline, interface_result
    