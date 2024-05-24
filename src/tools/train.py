
# 重新设置工作目录
import sys, os
sys.path[0] = os.path.join(sys.path[0], "../..")
from src.data import build_dataloader
from configs import load_arg
from src.model import build_model
from src.solver import build_optimizer, wrapper_lr_scheduler
import torch
from src.utils import get_free_device_ids
from datetime import datetime
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os.path as osp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


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
    cfg['save_dir'] = os.path.join(cfg['save_dir'], time_str + cfg['tag'])
    cfg['log_dir'] = os.path.join(cfg['log_dir'], time_str + cfg['tag'])
    if not os.path.isdir(cfg['save_dir']): os.makedirs(cfg['save_dir'])
    if not os.path.isdir(cfg['log_dir'] ): os.makedirs(cfg['log_dir'] )
    print("Save Directory :",cfg['save_dir'])
    print("Log Directory :", cfg['log_dir'])

    # 构建训练组件
    train_dataloader = build_dataloader(cfg['train_pipeline'])
    model = build_model(cfg['model'], pretrain_path=cfg['load_path'])
    loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
    optimizer = build_optimizer(cfg['optimizer'],model)
    lr_scheduler = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer)

    # 获取训练显卡
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
    # 开始训练

    model.cuda(master_device)
    scaler = GradScaler()
    pbar = tqdm(total=len(train_dataloader))
    writer = SummaryWriter(cfg['log_dir'])
    device = torch.device('cuda', master_device)
    
    for epoch in range(1, cfg['max_epochs']+1):

        loss_record = 0

        # 开始训练
        for batch_idx, samples in enumerate(train_dataloader):

            # 清除梯度
            optimizer.zero_grad()

            # 获取数据
            images = samples['image'].to(device, torch.float)
            masks = samples['mask'].to(device, torch.long)

            # 前向传播（使用自动混合精度训练）
            with autocast():
                output = model(images)
                loss = loss_fn(output, masks)

            # 反向传播
            scaler.scale(loss).backward()
            loss_record += loss.item()

            # 更新优化器
            scaler.step(optimizer)
            scaler.update()

            # End Per Banch
            pbar.update(1)
            
        # 调整策略
        lr_scheduler.EPOCH_COMPLETED()

        # 输出训练信息
        loss_record = loss_record / len(train_dataloader)
        pbar.write("Epoch [%d] SegLoss: %.4f" % (epoch, loss_record))
        pbar.reset()

        # 输出 TensorBoard 日志
        writer.add_scalar('Loss', loss_record, epoch)

        # 保存权重
        if epoch%1 == 0:
            torch.save(
                model.state_dict(), 
                osp.join(cfg['save_dir'], 'temp.pth'))

        if epoch%int(cfg['save_period'])==0:
            torch.save(
                model.state_dict(), 
                osp.join(cfg['save_dir'], f'{str(epoch)}.pth'))






    
    
