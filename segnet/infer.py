import os
import matplotlib.pyplot as plt
import numpy as np

max_epochs = 50
val_interval = 1
VAL_AMP = True
from .train import inference
from .dataloader import ConvertToMultiChannelBasedOnBratsClassesd_2020
import torch
from monai import data
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
from Segformer3d.dataloader import  ConvertBackToOriginalClasses


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


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


def segresnet_get_predict(segresnet, casename, model_name):

    root_dir = "dataset"
    path_dict = {'image': [], 'label': []}
    data_path = os.path.join(root_dir, casename)

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if not file_name.endswith('seg.nii'):
            path_dict['image'].append(file_path)
        else:
            path_dict['label'].append(file_path)

    validation_files = [path_dict]
    val_ds = data.Dataset(data=validation_files, transform=val_transform)

    segresnet.eval()
    with torch.no_grad():
        # select one image to evaluate and visualize the model output
        val_input = val_ds[0]["image"].unsqueeze(0).to(device)
        val_output = inference(segresnet, val_input)
        val_output = post_trans(val_output[0])

    image = val_ds[0]["image"].numpy()
    gt = val_ds[0]["label"].numpy()
    pred = val_output.detach().numpy()

    if model_name =='segresnet_origin':
        layer1 = pred[0]
        layer3 = pred[2]
        pred[2] = layer1 - layer3

    convert_back = ConvertBackToOriginalClasses()
    label_back = convert_back(gt)
    pred_back = convert_back(pred)

    return image, gt, pred, label_back, pred_back



if __name__ =="__main__":

    # model = SegResNet(
    #     blocks_down=[1, 2, 2, 4],
    #     blocks_up=[1, 1, 1],
    #     init_filters=16,
    #     in_channels=4,
    #     out_channels=3,
    #     dropout_prob=0.2,
    # ).to(device)
    #
    # model.load_state_dict(
    #     torch.load("segnet/pretrained/segresnet_model_origin.pth", map_location=device, weights_only=True)
    # )
    #
    #
    # image, gt, pred, label_back, pred_back = segresnet_get_predict(model, 'Brats17_2013_10_1', 'segresnet_origin')
    # print(image.shape)
    # print(gt.shape)
    # print(pred.shape)
    # print(label_back.shape)
    # print(pred_back.shape)
    # print(np.unique(label_back))
    # print(np.unique(pred_back))
    #
    #
    #
    # plt.subplot(121)
    # plt.imshow(label_back[..., 70])
    # plt.subplot(122)
    # plt.imshow(pred_back[..., 70])
    # plt.show()
    import nibabel as nib
    image = nib.load('dataset/Brats17_2013_10_1/Brats17_2013_10_1_flair.nii').get_fdata()
    print(image.shape)
