

# part                         26155, 25468
# cropland                     6063,  4587
# 8cm_sun1                     41400, 27600
# 8cm_sun2                     34300, 22498
# 8cm_cloud2                   34300, 22498
# 8cm_after2                   34294, 22496

# Jianli_8cm【46611, 31492】
# Hainan_A3_4cm_train【46611, 31492】
# Hainan_A3_4cm_test【24428, 22778】
# Hainan_A5_2cm_train【31539, 33168】
# Hainan_A5_2cm_test【20681, 19842】

config = dict(

    # Basic Config
    enable_backends_cudnn_benchmark = True,
    out_dir = r"out",
    max_num_devices = 1, 

    # Inference Dataset
    inference_pipeline = dict(

        shape = (24428, 22778),

        dataloader = dict(
            batch_size = 64,
            num_workers = 16,
            drop_last = True,
            pin_memory = True,
            shuffle = False,
            prefetch_factor = 4),

        dataset = dict(
            type="InferenceDataset",
            csv_file=r'/home/featurize/work/code/dataset/Hainan_A3_4cm_test.csv',
            image_dir=r'/home/featurize/data/TransferDataset/image/'),

        transforms = []
    ),

    # generator Network
    generator = dict(

        net = dict(
            type="deeplabv3plus_mcdda",
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

    # Classifier Network
    classifier = dict(

        net = dict(
            type="DRNSegPixelClassifier",
            num_classes=2,
            use_torch_up=False),
            
    ),

)