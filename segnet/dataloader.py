import os
import random
import glob
import torch
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai import data


root_dir = 'segnet/dataset'

""" - Brats 2017 dataset:
    + Annotations comprise the GD-enhancing tumor (ET — label 4)
    + The peritumoral edema (ED — label 2)
    + The necrotic and non-enhancing tumor (NCR/NET — label 1), 
    
    -Brats 2020 dataset:
    + Annotations comprise the GD-enhancing tumor (ET — label 4)
    + The peritumoral edema (ED — label 2)
    + The necrotic and non-enhancing tumor core (NCR/NET — label 1),
"""
def get_path(train_dir, test_dir):
    train_paths = []
    test_paths = []
    train_folder = glob.glob(train_dir + '/BraTS20_Training_*')
    test_folder = glob.glob(test_dir + '/BraTS20_Validation_*')

    for image_folder in train_folder:
        image_names = os.listdir(image_folder)
        label_path = glob.glob(image_folder + '/*eg*')[0]

        path_lists = []
        for image_name in image_names:
            image_path = os.path.join(image_folder, image_name)
            if image_path.endswith('seg.nii') or image_path.endswith("Segm.nii"):
                continue
            path_lists.append(image_path)
        path_lists.append(label_path)
        train_paths.append(path_lists)

    for image_folder in test_folder:
        image_names = os.listdir(image_folder)

        path_lists = []
        for image_name in image_names:
            image_path = os.path.join(image_folder, image_name)
            path_lists.append(image_path)
        test_paths.append(path_lists)

    return train_paths, test_paths


def datafold_read(data_path_dict):
    """
    data_path_dict: {'train': [295 list], 'val': [74 list]} each list [path1, path2, path3, path4, label_path]
    return:
    train: {'image: [path1, path2, path3, path4], 'label': [path]''}
    train: {'image: [path1, path2, path3, path4], 'label': [path]''}
    """
    train = []
    val =[]
    train_paths = data_path_dict['train']
    val_paths =  data_path_dict['val']
    for path in train_paths:
        folder = {}
        folder['image']= path[:-1]
        folder['label'] = path[-1]
        train.append(folder)
    for path in val_paths:
        folder = {}
        folder['image']= path[:-1]
        folder['label'] = path[-1]
        val.append(folder)
    return train, val


class ConvertToMultiChannelBasedOnBratsClassesd_2020(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 4 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 4 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 4))
            # label 4 is ET
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d

class ConvertToMultiChannelBasedOnBratsClassesd_2017(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    --> false:
    2: NET
    3: ET

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 3)
            d[key] = torch.stack(result, axis=0).float()
        return d


def get_dataloader2020(batch_size, data_path_dict):
    train_files, validation_files = datafold_read(data_path_dict)
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_2020(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_2020(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    val_ds = data.Dataset(data=validation_files, transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_ds, val_ds, train_loader, val_loader

if __name__ == "__name__":
    train_dir = 'segnet/dataset/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_dir = 'segnet/dataset/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

    train_val_paths, test_paths = get_path(train_dir, test_dir)
    print(len(train_val_paths))
    print(len(test_paths))

    random.shuffle(train_val_paths)
    train_size = int(0.8 * len(train_val_paths))  # 8/10 của tổng số

    train_paths = train_val_paths[:train_size]
    val_paths = train_val_paths[train_size:]

    data_path_dict = {'train': train_paths, 'val': val_paths}
    print(len(train_paths))
    print(len(val_paths))

