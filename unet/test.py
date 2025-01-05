"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    - Testing a pretrained model
    Hua Wang
"""
from unet.util import *
from unet.model import *
from unet.mylib import *
from unet.train import validation


def run_test(option):
    """
    Use model to predict the test dataset and calculate iou score & dice score
    :param option: pretrained model name, string
    :return:
    """
    # read data from csv
    test_dirs = pd.read_csv("../data/image_dirs/test_data.csv")

    # load my pretrained1 model
    unet = Unet(3)
    model_params = torch.load(option, map_location=torch.device('cpu'))
    unet.load_state_dict(model_params['model'])
    print("model loaded\nplease wait...")

    # make prediction with pretrained1 unet
    pred_dice_score, pred_iou_score = validation(
        unet, test_dirs, torch.device('cpu'), batch_size=393
    )  # 393 is the size of the test dataset
    print("IoU score: %s\nF1 score: %s" %
          (pred_iou_score.item(), pred_dice_score.item()))


if __name__ == '__main__':
    model = "pretrained5"
    run_test(f"../data/{model}/{model}.pth")
