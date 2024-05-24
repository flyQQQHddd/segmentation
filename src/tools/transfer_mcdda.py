


import sys, os
sys.path[0] = os.path.join(sys.path[0], "../..")
from itertools import cycle
from src.utils import get_free_device_ids
from datetime import datetime
import argparse
import torch
from tqdm import tqdm
from src.solver import wrapper_lr_scheduler
from src.data import build_dataloader
from src.solver.optimizer import build_optimizer
from src.model import build_model
import torch.nn.functional as F
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

#设置参数
def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--config_file",type=str,help="Path to config file")
    parser.add_argument("--load_path", type=str, help='Path of pretrained model')
    parser.add_argument("--tag", type=str, help='Tag of experience')
    parser.add_argument("--num_k", type=int, help='num_k')
    parser.add_argument("--merge_target_label", type=bool , help='是否融合目标标签')
    return parser.parse_args()

if __name__=="__main__":
    
    args = get_arguments()

    # 获取配置文件
    cfg = None
    config_file = args.config_file
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')
    exec(f"from {config_file} import config as cfg")

    # 是否融合目标标签
    print('Merge Target Label:', args.merge_target_label)

    # 创建输出目录
    now = datetime.now()
    time_str = now.strftime(r'%Y%m%d%H%M%S_')
    cfg['save_dir'] = os.path.join(cfg['save_dir'], time_str + args.tag)
    cfg['log_dir'] = os.path.join(cfg['log_dir'], time_str + args.tag)

    # 创建目录
    if not os.path.isdir(cfg['save_dir']): os.makedirs(cfg['save_dir'])
    if not os.path.isdir(cfg['log_dir'] ): os.makedirs(cfg['log_dir'] )
    print("Save Directory :",cfg['save_dir'])
    print("Log Directory :", cfg['log_dir'])

    # 构建生成器模型
    model_G = build_model(cfg['generator'], args.load_path)
    model_F1 = build_model(cfg['classifier'])
    model_F2 = build_model(cfg['classifier'])

    # 创建优化器
    optimizer_G = build_optimizer(cfg['optimizer'],model_G)
    optimizer_F1 = build_optimizer(cfg['optimizer'],model_F1)
    optimizer_F2 = build_optimizer(cfg['optimizer'],model_F2)

    # 构建调整策略
    lr_scheduler_G = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer_G)
    lr_scheduler_F1 = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer_F1)
    lr_scheduler_F2 = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer_F2)

    # 创建数据集
    source_dataloader = build_dataloader(cfg['source_pipeline'])
    target_dataloader = build_dataloader(cfg['target_pipeline'])
    if args.merge_target_label:
        target_labeled_dataloader = build_dataloader(cfg['target_pipeline_merge_label'])

    # 创建损失函数
    seg_loss_fn = torch.nn.CrossEntropyLoss()
    l1_loss_fn = torch.nn.L1Loss()

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

    model_G.cuda(master_device).train()
    model_F1.cuda(master_device).train()
    model_F2.cuda(master_device).train()
    pbar = tqdm(total=len_banch)
    writer = SummaryWriter(cfg['log_dir'])
    device = torch.device('cuda', master_device)

    for epoch in range(1, cfg['max_epochs']+1):

        # 重新获取训练数据
        trainloader_iter = enumerate(source_dataloader)
        targetloader_iter = enumerate(target_dataloader)
        if args.merge_target_label:
            target_labeled_loader_iter = cycle(enumerate(target_labeled_dataloader))

        # 损失函数值记录
        seg_loss_value = 0
        dis_loss_value2 = 0
        dis_loss_value3 = 0
        if args.merge_target_label:
            seg_loss_target_value = 0

        for i in range(len_banch):

            # 源域和目标域数据
            try:
                _, batch_source = next(trainloader_iter)
                _, batch_target = next(targetloader_iter)
                if args.merge_target_label:
                    _, batch_target_labeled = next(target_labeled_loader_iter)
            except:
                
                print('Loading data error')
                break

            # 获取源域数据
            images_source = batch_source['image'].to(device, torch.float)
            labels_source = batch_source['mask'].to(device, torch.long)
            # Step 1: 训练两个分类器
            # 使用源域数据进行训练，计算分割损失
            # 前向传播
            outputs = model_G(images_source)
            outputs1 = model_F1(outputs)
            outputs2 = model_F2(outputs)
            # 计算 loss
            seg_loss = seg_loss_fn(outputs1, labels_source) + seg_loss_fn(outputs2, labels_source)
            # 反向传播
            optimizer_G.zero_grad()
            optimizer_F1.zero_grad()
            optimizer_F2.zero_grad()
            seg_loss.backward()
            # 更新参数，更新生成器和分类器
            optimizer_G.step()
            optimizer_F1.step()
            optimizer_F2.step()
            # 记录损失，释放图
            seg_loss_value += seg_loss.item()


            # 获取目标域数据
            images_target = batch_target['image'].to(device, torch.float)
            # Step 2: 固定生成器，最大化分类器分歧
            # 使用源域和目标域数据进行训练
            feature_source = model_G(images_source).detach()
            pred_source1 = model_F1(feature_source)
            pred_source2 = model_F2(feature_source)
            feature_target = model_G(images_target).detach()
            pred_target1 = model_F1(feature_target)
            pred_target2 = model_F2(feature_target)
            # 计算 loss
            dis_loss = seg_loss_fn(pred_source1, labels_source) + seg_loss_fn(pred_source2, labels_source)
            dis_loss -= l1_loss_fn( 
                F.softmax(pred_target1, dim=1),
                F.softmax(pred_target2, dim=1))
            # 反向传播
            optimizer_F1.zero_grad()
            optimizer_F2.zero_grad()
            dis_loss.backward()
            # 更新参数，只更新分类器
            optimizer_F1.step()
            optimizer_F2.step()
            # 记录损失，释放图
            dis_loss_value2 += dis_loss.item()


            # Step 3: 固定分类器，训练生成器，最小化分类器分歧
            dis_loss_per_num = 0
            for i in range(args.num_k):
                # 前向传播
                feature_target = model_G(images_target)
                pred_target1 = model_F1(feature_target)
                pred_target2 = model_F2(feature_target)
                # 计算 loss
                dis_loss = l1_loss_fn(
                    F.softmax(pred_target1, dim=1), 
                    F.softmax(pred_target2, dim=1))
                # 反向传播
                optimizer_G.zero_grad()
                dis_loss.backward()
                # 更新参数，只更新生成器
                optimizer_G.step()
                # 记录损失，释放图
                dis_loss_per_num += dis_loss.item()
            dis_loss_value3 += dis_loss_per_num / args.num_k


            # Step 4: 使用带有标签的目标域数据进行训练（可选）
            if args.merge_target_label:
                # 获取带有标签的目标域数据image
                images_target_labeled = batch_target_labeled['image'].to(device, torch.float)
                labels_target_labeled = batch_target_labeled['mask'].to(device, torch.long)
                # 前向传播
                outputs = model_G(images_target_labeled)
                outputs1 = model_F1(outputs)
                outputs2 = model_F2(outputs)
                # 计算 loss
                seg_loss = seg_loss_fn(outputs1, labels_target_labeled)
                seg_loss += seg_loss_fn(outputs2, labels_target_labeled)
                # 反向传播
                optimizer_G.zero_grad()
                optimizer_F1.zero_grad()
                optimizer_F2.zero_grad()
                seg_loss.backward()
                # 更新参数，更新生成器和分类器
                optimizer_G.step()
                optimizer_F1.step()
                optimizer_F2.step()
                # 记录损失，释放图
                seg_loss_target_value = seg_loss.item()

            # End Per Batch
            pbar.update(1)

        # 调整策略
        lr_scheduler_G.EPOCH_COMPLETED()
        lr_scheduler_F1.EPOCH_COMPLETED()
        lr_scheduler_F2.EPOCH_COMPLETED()

        # 归一化损失值
        seg_loss_value = seg_loss_value / len_banch
        dis_loss_value2 = dis_loss_value2 / len_banch
        dis_loss_value3 = dis_loss_value3 / len_banch
        if args.merge_target_label:
            seg_loss_target_value = seg_loss_target_value / len_banch
   

        # 输出训练信息
        if args.merge_target_label:
            tqdm.write(
                "Epoch [%d] SegLoss: %.4f DisLoss2: %.4f DisLoss3: %.4f SegLossT: %.4f" % 
                (epoch, seg_loss_value, dis_loss_value2, dis_loss_value3, seg_loss_target_value))
        else:
            tqdm.write(
                "Epoch [%d] SegLoss: %.4f DisLoss2: %.4f DisLoss3: %.4f" % 
                (epoch, seg_loss_value, dis_loss_value2, dis_loss_value3))



        # 输出 TensorBoard 日志
        writer.add_scalar('SegLoss', seg_loss_value, epoch)
        writer.add_scalar('DisLoss2', dis_loss_value2, epoch)
        writer.add_scalar('DisLoss3', dis_loss_value3, epoch)

        if args.merge_target_label:

            writer.add_scalar('SegLossT', seg_loss_target_value, epoch)

            writer.add_scalars('All', {
                'SegLoss': seg_loss_value,
                'DisLoss2': dis_loss_value2,
                'DisLoss3': dis_loss_value2,
                'SegLossT': seg_loss_target_value
            }, epoch)

        else:

            writer.add_scalars('All', {
                'SegLoss': seg_loss_value,
                'DisLoss2': dis_loss_value2,
                'DisLoss3': dis_loss_value2
            }, epoch)





        # 保存权重
        if epoch%1 == 0:
            torch.save(model_G.state_dict(), osp.join(cfg['save_dir'], 'temp_G.pth'))
            torch.save(model_F1.state_dict(), osp.join(cfg['save_dir'], 'temp_F1.pth'))
            torch.save(model_F2.state_dict(), osp.join(cfg['save_dir'], 'temp_F2.pth'))
        if epoch%int(cfg['save_period'])==0:
            torch.save(model_G.state_dict(), osp.join(cfg['save_dir'], f'{str(epoch)}_G.pth'))
            torch.save(model_F1.state_dict(), osp.join(cfg['save_dir'], f'{str(epoch)}_F1.pth'))
            torch.save(model_F2.state_dict(), osp.join(cfg['save_dir'], f'{str(epoch)}_F2.pth'))

        # End Per Epoch
        pbar.reset()
            
