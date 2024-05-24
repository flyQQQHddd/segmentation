
# 重新设置工作目录
import sys, os
sys.path[0] = os.path.join(sys.path[0], "../..")

from src.data import build_dataloader
from src.model import build_model
from src.utils import create_zeros_png
from datetime import datetime
from argparse import ArgumentParser
from torch.cuda.amp import autocast
from PIL import Image
from tqdm import tqdm
import warnings
import torch
warnings.filterwarnings('ignore')

Image.MAX_IMAGE_PIXELS = 1000000000000000000

def load_arg():

    parser = ArgumentParser(description="MCDDA Inference")
    parser.add_argument("--config_file", type=str, help="Path to config file")
    parser.add_argument("--load_path_G", type=str, help='Path of pretrained model')
    parser.add_argument("--load_path_F", type=str, help='Path of pretrained model')
    parser.add_argument("--tag", type=str, help='Tag of experience')

    return parser.parse_args()

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
    generator = build_model(cfg['generator'], pretrain_path=arg['load_path_G']).eval()
    classifier = build_model(cfg['classifier'], pretrain_path=arg['load_path_F']).eval()
    image_pipeline = build_dataloader(cfg['inference_pipeline'])
    
    # 设置推理显卡
    master_device = 0

    # 模型结构稳定可以加快训练速度
    if cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True

    # -------------------------------------------------------------------
    # 开始推理

    generator = generator.cuda(master_device).eval()
    classifier = classifier.cuda(master_device).eval()
    image_w, image_h = cfg['inference_pipeline']['shape']
    predict_png = create_zeros_png(image_w, image_h)

    with torch.no_grad():

        for (image, pos_list) in tqdm(image_pipeline):

            image = image.float().cuda(master_device) 

            # 前向传播
            with autocast():
                feature = generator(image)
                predict = classifier(feature)
            
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

    # 输出推理结果
    pil_image = Image.fromarray(predict_png)
    pil_image.save(os.path.join(cfg['out_dir'], "output.png"))

    # End main
    del image_pipeline, predict_png
    