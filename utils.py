import torch
from monai.networks.nets import SegResNet
from Segformer3d.model import build_segformer3d_model, model_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name: str):
    segresnet = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    if model_name == 'segformer3d':
        segformer3D = build_segformer3d_model(model_config)
        segformer3D.load_state_dict(
            torch.load('Segformer3d/pretrain_model/segformer3d.pth', map_location=device, weights_only=True))
        return segformer3D

    elif model_name == 'segresnet_1':

        segresnet.load_state_dict(
            torch.load("segnet/pretrained/segresnet_model_1.pth", map_location=device, weights_only=True)
        )
        return segresnet
    elif model_name == 'segresnet_2':

        segresnet.load_state_dict(
            torch.load("segnet/pretrained/segresnet_model_2.pth", map_location=device, weights_only=True)
        )
        return segresnet

    elif model_name == 'segresnet_origin':
        segresnet.load_state_dict(
            torch.load("segnet/pretrained/segresnet_model_origin.pth", map_location=device, weights_only=True)
        )
        return segresnet

    else:
        print('please select model')
