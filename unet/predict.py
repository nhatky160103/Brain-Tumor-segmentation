"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    - Make prediction with a pretrained model
    Hua Wang
"""

import numpy
import numpy as np
from PIL import Image
from unet.model import *
from unet.mylib import *
import torchvision.transforms as T


def predict(test_img: numpy.ndarray, device, unet: Unet):
    """
    The function predict the mask area with the given pretrained U-Net model
    :param test_img: np array. [H, W, channel]
    :param device: torch.device
    :param unet: Unet
    :return: tensor. [batch size, channel, H, W]
    """
    # load data into device
    test_content_tensor = torch.from_numpy(test_img).to(device)
    test_content_tensor = test_content_tensor.float().unsqueeze(0)
    test_content_tensor = test_content_tensor.permute(0, 3, 1, 2)

    unet.eval()  # close BatchNorm2d during testing
    # calculate prediction of validation image with no autograd mechanism
    with torch.no_grad():
        pred_mask_tensor = unet.forward(test_content_tensor)
        pred_mask_tensor = torch.round(pred_mask_tensor)

    return pred_mask_tensor


def main():
    # read an image from kaggle_3m
    image = Image.open(
        "../kaggle_3m/TCGA_CS_4943_20000902/TCGA_CS_4943_20000902_15.tif"
    )
    img_array = np.array(image) / 255  # normalize to range [0, 1]

    # define device
    device = torch.device('cpu')

    # import a pretrained model
    unet = Unet(3)
    model_params = torch.load(f"../data/pretrained1/pretrained1.pth",
                              map_location=torch.device('cpu'))
    unet.load_state_dict(model_params['model'])

    # make prediction
    pred_mask = predict(img_array, device, unet)

    # permute the tensor and convert to PIL Image
    transform = T.ToPILImage()
    pred_mask = pred_mask[0]  # from [1, 1, 256, 256] to [256, 256]
    pred_image = transform(pred_mask)

    # show original image and the predicted mask
    image.show()
    pred_image.show()


if __name__ == '__main__':
    main()
