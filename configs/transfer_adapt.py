'''
迁移学习训练配置文件
DeepLab V3++
Resnet50
StepLR
Adam
'''

import albumentations as A

config = dict(
    
    # Basic Config
    enable_backends_cudnn_benchmark = True,
    max_epochs = 45,
    save_period = 5,
    save_dir = r"checkpoints",
    log_dir = r"log",
    out_dir = r"out",
    multi_gpu = False,
    max_num_devices = 1, 

    # Source Dataset
    source_pipeline = dict(
        dataloader = dict(
            batch_size = 16,
            num_workers = 16,
            drop_last = True,
            pin_memory = True,
            shuffle = True,
            prefetch_factor = 2),

        dataset = dict(
            type="PNGDataset",
            csv_file=r'/home/featurize/work/code/dataset/JianLi_DOM_8cm_sun1.csv',
            image_dir=r'/home/featurize/data/WHUSDataset/image/',
            mask_dir=r'/home/featurize/data/WHUSDataset/label/'),

        transforms = [
            A.RandomCrop(512, 512, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(p=0.5)
        ]
    ),


    # Target Dataset
    target_pipeline = dict(
        dataloader = dict(
            batch_size = 16,
            num_workers = 16,
            drop_last = True,
            pin_memory = True,
            shuffle = True,
            prefetch_factor = 2),

        dataset = dict(
            type="PNGDataset",
            csv_file=r'/home/featurize/data/WHUSDataset/part.csv',
            image_dir=r'/home/featurize/data/WHUSDataset/image/'),

        transforms = [
            A.RandomCrop(512, 512, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(p=0.5)
        ]
    ),
        

    # Segmentation Network
    model = dict(

        net = dict(
            type="Deeplabv3plus",
            num_classes=3),

        backbone = dict(
            type="resnet50",
            pretrained=True,
            replace_stride_with_dilation=[False,False,2]),

        head = dict(
            type="ASPP",
            in_channels=2048,
            out_channels=256,
            dilation_list=[6,12,18]),
    ),
    

    # Discriminator Network
    discriminator = dict(

        net = dict(
            type="FCDiscriminator",
            num_classes=3,
            ndf=64),
    ),


    # Solver
    lr_scheduler = dict(type="StepLR",step_size=5,gamma=1/3), 
    optimizer = dict(type="Adam",lr=2.5e-4,betas=(0.9, 0.99)),
)