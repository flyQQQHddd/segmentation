import albumentations as A

config = dict(

    # Basic Config
    enable_backends_cudnn_benchmark = True,
    max_epochs = 45,
    save_period = 5,
    save_dir = r"checkpoints",
    log_dir = r"log",
    multi_gpu = False,
    max_num_devices = 1, 

    # Train Dataset
    train_pipeline = dict(
        
        dataloader = dict(
            batch_size = 16,
            num_workers = 16,
            drop_last = True,
            pin_memory = True,
            shuffle = True,
            prefetch_factor = 4),

        dataset = dict(
            type="PNGDataset",
            csv_file=r'/home/featurize/work/code/dataset/A5N05.csv',
            image_dir=r'/home/featurize/data/TransferDataset/image/',
            mask_dir=r'/home/featurize/data/TransferDataset/label/'),

        transforms = [
            A.RandomCrop(512, 512, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(p=0.5)
        ]
    ),

    # Model
    model = dict(

        net = dict(
            type="Deeplabv3plus",
            num_classes=2),

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

    # Solver
    lr_scheduler = dict(
        type="StepLR",
        step_size=5,
        gamma=1/3
    ), 

    optimizer = dict(
        type="Adam",
        lr=1e-4,
        weight_decay=1e-5
    ),
)