import os
from monai.data import CacheDataset, list_data_collate, DataLoader
from monai.transforms import (
    LoadImaged,
    RandCropByPosNegLabeld,
    ToTensord,
    Compose,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandAffined,
    CropForegroundd,
)
from .data_handler import read_json_file


def add_root_dir(data_list, root_dir):
    """
    Adds a root directory to the 'image' and 'label' paths in a list of dictionaries.

    :param data_list: List of dictionaries containing 'image' and 'label' keys.
    :param root_dir: Root directory to prepend to the paths.
    :return: Updated list of dictionaries with modified paths.
    """
    for item in data_list:
        if 'image' in item:
            item['image'] = os.path.join(root_dir, item['image'])
        if 'label' in item:
            item['label'] = os.path.join(root_dir, item['label'])
    return data_list
        

def get_segmentation_dataloaders(config, percent):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'decathlon_pancreas_splits.json') 
    datadirs = read_json_file(file_path)
    train_dirs = datadirs[f"train_{percent}_percent"]
    val_dirs = datadirs['validation']
    train_datalist = add_root_dir(train_dirs, config['data_root_dir'])
    val_datalist = add_root_dir(val_dirs, config['data_root_dir'])
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config['spacing'], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=config["min_intensity"], a_max=config["max_intensity"], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        RandAffined(
                keys=["image", "label"],
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                prob=0.1
            ),
        RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=config['roi_size'],
                    pos=1,
                    neg=1,
                    num_samples=config['batch_size']//2,
                    image_key="image",
                    image_threshold=0,
                ),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config['spacing'], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=config["min_intensity"], a_max=config["max_intensity"], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])

    # we use cached datasets - these are 10x faster than regular datasets
    train_ds = CacheDataset(
        data=train_datalist,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=1,
    )
    val_ds = CacheDataset(
        data=val_datalist,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)

    # train_ds = Dataset(data=train_datalist, transform=train_transforms)
    # val_ds = Dataset(data=val_datalist, transform=val_transforms)
    # train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1)
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    return train_loader, val_loader


