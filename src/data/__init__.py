

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from albumentations.pytorch import ToTensorV2
from albumentations import Compose
from . import dataset as Datasets

from copy import deepcopy

def build_dataloader(cfg_data_pipeline: dict) -> DataLoader:
    """Build dataloader by config

    Args:
        cfg_data_pipeline (dict): 

    Returns:
        dataloader: torch.utils.data.DataLoader
    """
    cfg_data_pipeline = deepcopy(cfg_data_pipeline)
    cfg_dataset = cfg_data_pipeline.pop('dataset')
    cfg_transforms = cfg_data_pipeline.pop('transforms')
    cfg_dataloader = cfg_data_pipeline.pop('dataloader')

    transforms = build_transforms(cfg_transforms)
    dataset = build_dataset(cfg_dataset,transforms)

    # 构建数据加载器
    dataloader  = DataLoader(dataset, **cfg_dataloader)
    return dataloader


def build_dataset(cfg_dataset:dict, transforms=None) -> Dataset:
    '''Build dataset by config

    Args (type):
        cfg_dataset (dict):
        transforms (callable,optional): Optional transforms to be applied on a sample.
    return:
        dataset(torch.utils.data.Dataset)
    '''

    dataset_type = cfg_dataset.pop('type')
    dataset_kwags = cfg_dataset

    if hasattr(Datasets,dataset_type):
        dataset = getattr(Datasets,dataset_type)(**dataset_kwags,transforms=transforms)
    else:
        raise ValueError("\'type\' of dataset is not defined. Got {}".format(dataset_type))

    return dataset


def build_transforms(cfg_transforms:list):
    '''Build transforms by config

    Args (type):
        cfg_transforms (list):
    return:
    '''

    cfg_transforms.append(ToTensorV2())

    return Compose(cfg_transforms)

