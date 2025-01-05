"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    -Training and validating a U-Net model
    Hua Wang
"""

from sklearn.model_selection import train_test_split
from unet.model import *
from unet.util import *
import torch


def validation(unet: Unet, validation_img: pd.DataFrame, device,
               batch_size: int = 10):
    """
    Make predictions on the validation/test dataset given a U-Net
    model and file directories to read the dataset

    :param unet: Unet, a pretrained model
    :param validation_img: dataframe, file directories for data set
    :param device: torch.device
    :param batch_size: int
    :return: pred_dice_score, pred_iou_score, tensors
    """

    # load data into device
    validation_content, validation_mask = load_data(
        validation_img, shuffle=True, batch_size=batch_size
    )
    validation_content, validation_mask = adjust_data(
        validation_content, validation_mask
    )
    validation_content_tensor = \
        torch.from_numpy(validation_content).to(device).float()
    validation_mask_tensor = \
        torch.from_numpy(validation_mask).to(device).float()

    unet.eval()  # close BatchNorm2d during validation

    # calculate prediction of validation image with no autograd mechanism
    with torch.no_grad():
        pred_mask_tensor = unet.forward(validation_content_tensor)
        pred_mask_tensor = torch.round(pred_mask_tensor)
        pred_dice_score = dice_score(pred_mask_tensor, validation_mask_tensor)
        pred_iou_score = iou_score(pred_mask_tensor, validation_mask_tensor)

    return pred_dice_score, pred_iou_score


def train(train_img: pd.DataFrame,
          validation_img: pd.DataFrame,
          lr: float = 0.0001,
          use_cel: bool = True,
          epoch: int = 1,
          shuffle: bool = False,
          batch_size: int = 32,
          device_: str = 'cpu'):
    """
    Train the model based on training set, given file directories to load
    the training set and validation set
    :param train_img: dataframe, file directories for training data
    :param validation_img: dataframe, file directories for validation data
    :param lr: float, learning rate
    :param use_cel: bool, whether to use cross entropy loss
    :param epoch: int
    :param shuffle: bool
    :param batch_size: int
    :param device_: string
    :return: none
    """
    # define device
    if device_ != 'cuda' and device_ != 'cpu':
        raise ValueError("invalid device")
    if not torch.cuda.is_available() and device_ == 'cuda':
        device = torch.device('cpu')
    else:
        device = torch.device(device_)
    print("using", device.type)

    # how many iterations do we need to go through all data in an epoch
    iteration = train_img.shape[0] // batch_size

    # execute validation on every 20 iteration
    interval = 20

    # initialize model and model parameters
    print("initialize model")
    unet = Unet(3).to(device)  # 3 RGB channels
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dice_loss = MyLoss(dice_loss_mode=True, smooth=0.01)

    for e in range(epoch):
        for i in range(iteration):
            # get train data_analysis and masks (we drop last few samples)
            train_content, train_mask = load_data(train_img,
                                                  start_idx=i * batch_size,
                                                  shuffle=shuffle,
                                                  batch_size=batch_size)

            # rescaled content data and convert mask data into binary digit
            train_content, train_mask = adjust_data(train_content, train_mask)

            # create tensor matrix for model to train
            train_content_tensor = torch.from_numpy(train_content).to(device)
            train_mask_tensor = torch.from_numpy(train_mask).to(device)

            # Now our data is double(float64 in pytorch),
            # while the weights in conv are float
            # So, convert our data to float32
            train_content_tensor = train_content_tensor.float()
            train_mask_tensor = train_mask_tensor.float()

            # train data
            # set the model in the training mode
            unet.train()
            optimizer.zero_grad()  # clear grad, avoid accumulation
            pred_mask_tensor = \
                unet.forward(train_content_tensor)  # get model prediction
            loss = dice_loss(pred_mask_tensor, train_mask_tensor)
            if use_cel:
                loss += criterion(pred_mask_tensor, train_mask_tensor)
            accuracy = dice_score(pred_mask_tensor, train_mask_tensor)
            loss.backward()  # backpropagation
            optimizer.step()  # update model weight

            # print training log
            print('epoch-%s-iteration-%s: loss %s accuracy %s'
                  % (e + 1, i + 1, loss.item(), accuracy.item()))
            if (i + 1) % interval == 0:
                valid_dice_score, valid_iou_score = \
                    validation(unet, validation_img, device,
                               batch_size=validation_img.shape[0])
                unet.train()  # set the mode back to train after validation
                print('valid iou score: %s valid dice score: %s' %
                      (valid_iou_score.item(), valid_dice_score.item()))
                # save model if valid_iou_score exceed 0.8
                if valid_iou_score.item() > 0.8:
                    torch.save({'model': unet.state_dict()},
                               'unet_epoch%s_iter%s.pth' % (e + 1, i + 1))


def main():
    # get user inputs
    split = int(
        input("Would you like to re-split the data (1 for yes, 0 for no)? "))
    print(
        "If you're using cuda, the recommended batch size would be 32 or 64.\n"
        + "It would take about 30min to train the model using cuda")
    print(
        "If you're using cpu, the recommended batch size would be 16.\n"
        + "It would take >2h to train the model on cpu."
    )
    device = input("Set device (cuda or cpu): ")
    if device == 'cpu':
        print("Recommend: 1 epoch, 16 batch size")
    epoch = int(input("Set epoch: "))
    batch_size = int(input("Set batch size: "))

    if split:
        root_dir = "../kaggle_3m"
        # select all file paths into two dataframes
        masks, contents = extract_paths(root_dir)
        # sort paths and combine dataframes
        dir_df = sort_combine_paths(masks, contents)
        # split training set, validation set and test set,
        # not depend on patient id
        train_dirs, test_dirs = train_test_split(
            dir_df, test_size=0.1, random_state=40
        )
        # we won't save these data frame
        # because we don't want to overwrite previous data sets.
    else:
        # read training data from previous saved csv file
        train_dirs = pd.read_csv("../data/image_dirs/train_data.csv")

    train_dirs, validation_dirs = train_test_split(
        train_dirs, test_size=0.2, random_state=40
    )

    # train
    train(train_img=train_dirs, validation_img=validation_dirs, epoch=epoch,
          batch_size=batch_size, device_=device)


if __name__ == '__main__':
    main()
