import sys, os
sys.path[0] = os.path.join(sys.path[0], "../..")

from configs import load_arg
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os.path as osp
from src.solver import wrapper_lr_scheduler
from src.data import build_dataloader
from src.solver.optimizer import build_optimizer
from src.model import build_model
from src.utils import get_free_device_ids
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    arg = vars(load_arg())
    lambda_adv_target = 0.01

    # 获取配置文件
    cfg = None
    config_file = arg["config_file"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')
    exec(f"from {config_file} import config as cfg")

    # 参数融合，优先级：arg > cfg
    for key, value in arg.items():
        cfg[key] = value

    # 创建输出目录
    now = datetime.now()
    time_str = now.strftime(r'%Y%m%d%H%M%S_')
    cfg['save_dir'] = os.path.join(cfg['save_dir'], time_str + cfg['tag'])
    cfg['log_dir'] = os.path.join(cfg['log_dir'], time_str + cfg['tag'])

    # 创建目录
    if not os.path.isdir(cfg['save_dir']): os.makedirs(cfg['save_dir'])
    if not os.path.isdir(cfg['log_dir'] ): os.makedirs(cfg['log_dir'] )
    print("Save Directory :",cfg['save_dir'])
    print("Log Directory :", cfg['log_dir'] )

    # 构建模型
    segmentation = build_model(cfg['model'], pretrain_path=cfg['load_path'])
    discriminator = build_model(cfg['discriminator']).train()

    # 创建优化器
    optimizer = build_optimizer(cfg['optimizer'],segmentation)
    optimizer_D = build_optimizer(cfg['optimizer'],discriminator)

    # 构建调整策略
    lr_scheduler = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer)
    lr_scheduler_D = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer_D)

    # 创建数据集
    source_dataloader = build_dataloader(cfg['source_pipeline'])
    target_dataloader = build_dataloader(cfg['target_pipeline'])

    # 创建损失函数
    mse_loss = torch.nn.MSELoss()
    cel_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    # labels for adversarial training
    source_label = 0
    target_label = 1

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
    
    # banch 的数量取源域数据与目标域数据中较少的一个
    len_source_loader = len(source_dataloader)
    len_target_loader = len(target_dataloader)
    len_banch = min([len_source_loader, len_target_loader])


    # -------------------------------------------------------------------
    # 开始训练
    segmentation.cuda(master_device)
    discriminator.cuda(master_device)
    pbar = tqdm(total=len_banch)
    writer = SummaryWriter(cfg['log_dir'])
    device = torch.device('cuda', master_device)

    for epoch in range(1, cfg['max_epochs']+1):

        # 重新获取训练数据
        trainloader_iter = enumerate(source_dataloader)
        targetloader_iter = enumerate(target_dataloader)

        for i in range(len_banch):

            loss_seg_record = 0
            loss_adv_G_record = 0
            loss_adv_D_record = 0

            # 源域和目标域数据
            try:
                _, batch_source = next(trainloader_iter)
                _, batch_target = next(targetloader_iter)
            except:
                break

            optimizer.zero_grad()
            optimizer_D.zero_grad()

            ##########################################
            # 训练生成器
            ## don't accumulate grads in D
            for param in discriminator.parameters():
                param.requires_grad = False

            ## 源域数据训练
            # 获取数据
            images = batch_source['image'].to(device, torch.float)
            labels = batch_source['mask'].to(device, torch.long)
            # 前向传播
            pred = segmentation(images)
            # 计算 LOSS
            loss_seg = cel_loss(pred, labels)
            # 反向传播
            loss_seg.backward()
            loss_seg_record += loss_seg.item()


            ## 目标域数据训练
            # 获取数据
            images = batch_target['image'].to(device, torch.float)
            # 前向传播
            pred_target = segmentation(images)
            D_out = discriminator(F.softmax(pred_target))
            # 计算 LOSS
            loss_adv_target = mse_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
            loss = lambda_adv_target * loss_adv_target
            # 反向传播
            loss.backward()
            loss_adv_G_record += loss_adv_target.item()

            # 更新参数
            optimizer.step()
        
            ##########################################
            # 训练判别器
            # bring back requires_grad
            for param in discriminator.parameters():
                param.requires_grad = True

            ## 源域数据训练
            pred = pred.detach()
            D_out = discriminator(F.softmax(pred))
            # 计算LOSS
            loss_D = mse_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
            loss_D = loss_D * 0.5
            # 反向传播
            loss_D.backward()
            loss_adv_D_record += loss_D.item()

            ## 目标域数据训练
            pred_target = pred_target.detach()
            D_out = discriminator(F.softmax(pred_target))
            # 计算损失LOSS
            loss_D = mse_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))
            loss_D = loss_D *  0.5
            # 反向传播
            loss_D.backward()
            loss_adv_D_record += loss_D.item()

            # 更新参数
            optimizer_D.step()

            # End Per Banch
            pbar.update(1)
            
            
        # 调整策略
        lr_scheduler.EPOCH_COMPLETED()
        lr_scheduler_D.EPOCH_COMPLETED()

        ## 输出训练信息
        tqdm.write("Epoch [%d] SegLoss: %.4f AdvLoss: %.4f DLoss: %.4f" % (epoch, loss_seg_record, loss_adv_G_record, loss_adv_D_record))

        # 输出 TensorBoard 日志
        writer.add_scalar('SegLoss', loss_seg_record, epoch)
        writer.add_scalar('AdvLoss', loss_adv_G_record, epoch)
        writer.add_scalar('DLoss', loss_adv_D_record, epoch)

        # 保存权重
        if epoch%1==0:
            torch.save(
                segmentation.state_dict(), 
                osp.join(cfg['save_dir'], 'temp.pth'))
            torch.save(
                discriminator.state_dict(), 
                osp.join(cfg['save_dir'], 'D_temp.pth'))

        if epoch%int(cfg['save_period'])==0:
            torch.save(
                segmentation.state_dict(), 
                osp.join(cfg['save_dir'], str(epoch) + '.pth'))
            torch.save(
                discriminator.state_dict(), 
                osp.join(cfg['save_dir'], str(epoch) + '_D.pth'))

        # End Per Epoch
        pbar.reset()

        

